import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Collaborative Filtering Recommender System.
 * Uses Alternating Least Squares (ALS) to generate product recommendations for users.
 */
object RecommenderSystem {
  def main(args: Array[String]): Unit = {

    // 1. Spark Session Initialization
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("AmazonRecommenderSystem")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("[INFO] --- STARTING RECOMMENDATION PIPELINE ---")

    // ---------------------------------------------------------
    // STEP 1: DATA INGESTION
    // ---------------------------------------------------------
    // Loading interactions (User, Item, Rating) from MongoDB.
    // Explicitly selecting 'userID' as the key identifier.
    val ratingsDF = spark.read
      .format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .option("database", "amazon_db")
      .option("collection", "reviews")
      .load()
      .select("userID", "productID", "overall")
      .na.drop()

    println(s"[INFO] Total rating interactions loaded: ${ratingsDF.count()}")

    // ---------------------------------------------------------
    // STEP 2: PREPROCESSING (NUMERIC INDEXING)
    // ---------------------------------------------------------
    // ALS requires numeric indices for users and items. We use StringIndexer to map strings to integers.
    val userIndexer = new StringIndexer()
      .setInputCol("userID")
      .setOutputCol("userIndex")

    val productIndexer = new StringIndexer()
      .setInputCol("productID")
      .setOutputCol("productIndex")

    // We use a Pipeline to ensure the transformation model is consistent.
    val indexerPipeline = new Pipeline().setStages(Array(userIndexer, productIndexer))

    // Fit the indexer model ONCE and reuse it. (Optimization over original code)
    val indexerModel = indexerPipeline.fit(ratingsDF)
    val indexedRatingsDF = indexerModel.transform(ratingsDF)

    // CHECKPOINTING: Crucial for ALS to prevent StackOverflowErrors on large lineage graphs.
    spark.sparkContext.setCheckpointDir("checkpoint_dir")

    // ---------------------------------------------------------
    // STEP 3: MODEL TRAINING
    // ---------------------------------------------------------
    // Splitting data: 80% for Training, 20% for Evaluation.
    val Array(trainingData, testData) = indexedRatingsDF.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Cache training data to speed up the iterative ALS process.
    trainingData.cache()

    println("[INFO] Training ALS Model...")

    val alsEstimator = new ALS()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setUserCol("userIndex")
      .setItemCol("productIndex")
      .setRatingCol("overall")
      .setColdStartStrategy("drop") // Prevents NaNs during evaluation
      .setCheckpointInterval(2)

    val alsModel = alsEstimator.fit(trainingData)
    println("[SUCCESS] Model trained successfully.")

    // Clear memory
    trainingData.unpersist()

    // ---------------------------------------------------------
    // STEP 4: EVALUATION
    // ---------------------------------------------------------
    println("[INFO] Evaluating model on test data...")
    val predictionsDF = alsModel.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("overall")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictionsDF)
    println(f"[RESULT] Root Mean Square Error (RMSE): $rmse%1.4f")

    // ---------------------------------------------------------
    // STEP 5: GENERATING RECOMMENDATIONS
    // ---------------------------------------------------------
    println("[INFO] Generating top 5 recommendations for a user subset...")

    // Select a distinct subset of users from the test set for demonstration
    val userSubsetDF = testData.select("userIndex").distinct().limit(50)

    // Generate raw recommendations (UserIndex -> [ProductIndex, Rating])
    val userRecsDF = alsModel.recommendForUserSubset(userSubsetDF, 5)

    // Explode the array to created rows: UserIndex, ProductIndex, Rating
    val explodedRecsDF = userRecsDF
      .withColumn("rec", explode($"recommendations"))
      .select($"userIndex", $"rec.productIndex", $"rec.rating")
      // Rating Clamping: Ensure predictions stay within valid 1.0-5.0 bounds
      .withColumn("rating",
          when($"rating" > 5.0, 5.0)
          .when($"rating" < 1.0, 1.0)
          .otherwise($"rating")
      )

    // ---------------------------------------------------------
    // STEP 6: POST-PROCESSING (DECODING IDs)
    // ---------------------------------------------------------
    // Retrieve the fitted StringIndexer models from Step 2 to map Indices back to Strings.
    val userIndexerModel = indexerModel.stages(0).asInstanceOf[StringIndexerModel]
    val productIndexerModel = indexerModel.stages(1).asInstanceOf[StringIndexerModel]

    val productConverter = new IndexToString()
      .setInputCol("productIndex")
      .setOutputCol("productID")
      .setLabels(productIndexerModel.labels)

    val userConverter = new IndexToString()
      .setInputCol("userIndex")
      .setOutputCol("userID") // Explicitly mapping back to 'userID'
      .setLabels(userIndexerModel.labels)

    // Apply the converters
    val readableRecsDF = productConverter.transform(
        userConverter.transform(explodedRecsDF)
      )

    // ---------------------------------------------------------
    // STEP 7: ENRICHMENT (JOINING METADATA)
    // ---------------------------------------------------------
    println("\n[INFO] Enriching recommendations with product titles from MongoDB...")

    val productFactorsDF = alsModel.itemFactors
    println(s"[INFO] Learned latent factors for ${productFactorsDF.count()} products.")

    // Load product metadata (Title, Price)
    val productsInfoDF = spark.read
      .format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .option("database", "amazon_db")
      .option("collection", "products")
      .load()
      .select("productID", "title", "price")

    // Join recommendations with metadata
    // Sorting by userID and then by Rating (descending)
    val richRecommendationsDF = readableRecsDF
      .join(productsInfoDF, "productID")
      .select($"userID", $"title", $"price", $"rating")
      .sort($"userID", $"rating".desc)

    println("--- TOP RECOMMENDATIONS WITH TITLES ---")
    richRecommendationsDF.show(10, truncate = false)

    spark.stop()
  }
}