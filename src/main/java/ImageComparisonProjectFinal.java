import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Result;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;
import java.io.File;
import java.util.*;

public class ImageComparisonProjectFinal {

    public static String inputImage; //path desired image to search for
    public static String imageFolder; //path to folder of images to compare to input
    public static SavedModelBundle model; //tensorflow model to compare features
    public static int numSimilar; //number of similar images you want to receive
    //See notes file for information on these values
    public static String inputTName; //specific name needed to pass to the model
    public static String outputTName; //specific name needed to pass to the model
    public static Size modelImageSize; //width and height needed for model to run


    public static void main(String[] args) {
        OpenCV.loadLocally();
        inputImage = "testImages/box.jpg";
        imageFolder = "testImages";
        model = SavedModelBundle.load("resnet-v2-tensorflow2-152-feature-vector-v2");
        inputTName = "serving_default_inputs";
        outputTName = "StatefulPartitionedCall";
        numSimilar = 10;
        TreeMap<Double, String> tree = findSimilarImages();
        //temp code, prints file and score
        int tmpIter = 0;
        for(Map.Entry<Double, String> entry : tree.entrySet()) {
            System.out.println("Filename: " + entry.getValue() + ", Score: " + entry.getKey());
            tmpIter++;
            if(tmpIter > numSimilar){
                break;
            }
        }
    }

    //Uses TreeMap to sort the images by similarity to the query image, returns top 5 as ArrayList
    private static TreeMap<Double, String> findSimilarImages() {
        TreeMap<Double, String> similarityMap = new TreeMap<>(Collections.reverseOrder());

        //check for folder
        File folder = new File(imageFolder);
        if (!folder.exists() || !folder.isDirectory()) {
            throw new IllegalArgumentException("Invalid folder path: " + imageFolder);
        }

        //check for jpg, iterate for every file in the folder, store result in treemap
        for (File file : Objects.requireNonNull(folder.listFiles())) {
            if (file.isFile() && file.getName().endsWith(".jpg")) {
                double similarity = compareImages(inputImage, file.getAbsolutePath());
                similarityMap.put(similarity, file.getAbsolutePath());
            }
        }

        return similarityMap;
    }

    //Does multiple image comparison methods, summarizes the results in a value 0-1
    private static double compareImages(String queryImage, String compImage) {
        //use OpenCV to convert to matrices
        Mat queryI = Imgcodecs.imread(queryImage);
        Mat compI = Imgcodecs.imread(compImage);

        double tmpWidth = Math.min(queryI.width(), compI.width());
        double tmpHeight = Math.min(queryI.height(), compI.height());
        modelImageSize = new Size(tmpWidth, tmpHeight);

        //Resize images
        Imgproc.resize(queryI, queryI, modelImageSize);
        Imgproc.resize(compI, compI, modelImageSize);

        //Uses TensorFlow model to get feature vectors, compares them
        float[] queryDeepFeatures = extractDeepFeatures(queryI);
        float[] compDeepFeatures = extractDeepFeatures(compI);
        Mat queryDeepMatrix = new Mat(1, queryDeepFeatures.length, CvType.CV_32F);
        Mat compDeepMatrix = new Mat(1, compDeepFeatures.length, CvType.CV_32F);
        queryDeepMatrix.put(0, 0, queryDeepFeatures);
        compDeepMatrix.put(0, 0, compDeepFeatures);
        //System.out.println(dotProductSimilarity(queryDeepMatrix, compDeepMatrix));
        //System.out.println(euclideanDistance(queryDeepMatrix, compDeepMatrix));
        //System.out.println(cosineSimilarity(queryDeepMatrix, compDeepMatrix));
        return cosineSimilarity(queryDeepMatrix, compDeepMatrix);
    }

    //Uses tensorflow model to get feature vectors for given image
    private static float[] extractDeepFeatures(Mat imageData) {
        try (Tensor inputTensor = preprocessImage(imageData)){
            try (Result outputTensor = model.session().runner()
                    .feed("serving_default_inputs", inputTensor)
                    .fetch("StatefulPartitionedCall")
                    .run()) {
                return StdArrays.array2dCopyOf((FloatNdArray) outputTensor.get(0))[0];
            } catch (Exception e) {
                e.printStackTrace();
                return new float[1];
            }
        } catch (Exception e) {
            e.printStackTrace();
            return new float[1];
        }
    }

    //Convert image data to Tensor, might be able to optimize this
    private static Tensor preprocessImage(Mat imageData) {
        imageData.convertTo(imageData, CvType.CV_32FC3); // Convert to float32
        Core.normalize(imageData, imageData, 0, 1, Core.NORM_MINMAX);
        float[] imgData = new float[(int) (imageData.total() * imageData.channels())];
        imageData.get(0, 0, imgData);
        int heightVal = (int) modelImageSize.height;
        int widthVal = (int) modelImageSize.width;

        int oldIter = 0;
        float[][][][] newImageData = new float[1][heightVal][widthVal][3];
        for(int i = 0; i < heightVal; i++){
            for(int j = 0; j < widthVal; j++){
                for(int k = 0; k < 3; k++){
                    newImageData[0][i][j][k] = imgData[oldIter];
                    oldIter++;
                }
            }
        }
        return TFloat32.tensorOf(StdArrays.ndCopyOf(newImageData));
    }

    //calculates cosine similarity of two matrices
    //normally in range -1 to 1, shifted to be 0-1
    private static double cosineSimilarity(Mat img1, Mat img2) {
        double dotProd =  dotProductSimilarity(img1, img2);
        double mag1 =  Core.norm(img1, Core.NORM_L2);
        double mag2 =  Core.norm(img2, Core.NORM_L2);
        if(mag1 != 0.0 && mag2 != 0.0){
            return ((dotProd / (mag1 * mag2)) + 1.0) / 2.0;
        }
        return 0.0;
    }

    private static double dotProductSimilarity(Mat img1, Mat img2) {
        return img1.dot(img2);
    }

    private static double euclideanDistance(Mat img1, Mat img2) {
        return Core.norm(img1, img2, Core.NORM_L2);
    }
}

