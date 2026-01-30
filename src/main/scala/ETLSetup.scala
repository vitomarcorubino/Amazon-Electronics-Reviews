import org.apache.spark.sql.{SparkSession, SaveMode, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * ETL Process for Amazon Electronics Dataset.
 * Handles ingestion from JSON, text normalization, and optimized persistence to MongoDB.
 */
object ETLSetup {
  def main(args: Array[String]): Unit = {

    // 1. Spark Session Configuration
    val spark = SparkSession.builder()
      .appName("AmazonElectronicsETL")
      .master("local[*]")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .config("spark.sql.caseSensitive", "true")
      .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("[INFO] Starting ETL pipeline execution with Broadcast Join optimization...")

    // ---------------------------------------------------------
    // STEP 1: INGESTION AND CLEANING OF REVIEWS
    // ---------------------------------------------------------
    val reviewsInputPath = "data/reviews_electronics.json"
    println(s"[INFO] Loading source reviews from: $reviewsInputPath")
    val rawReviewsDF = spark.read.json(reviewsInputPath)

    // Data Sampling: 5% fraction with a fixed seed for deterministic testing/benchmarking.
    val sampledReviewsDF = rawReviewsDF.sample(withReplacement = false, fraction = 0.05, seed = 42)

    // --- REVIEWS DATA TRANSFORMATION ---
    val cleanedReviewsDF = sampledReviewsDF
      // Convert Unix timestamp to Spark DateType
      .withColumn("reviewDate", to_date(from_unixtime($"unixReviewTime")))

      // ADVANCED TEXT NORMALIZATION: Sanitizing HTML entities and extra whitespace
      .withColumn("reviewText", regexp_replace($"reviewText", "<[^>]+>", " "))
      .withColumn("reviewText", regexp_replace($"reviewText", "&nbsp;", " "))
      .withColumn("reviewText", regexp_replace($"reviewText", "&amp;", "&"))
      .withColumn("reviewText", regexp_replace($"reviewText", "&quot;", "\""))
      .withColumn("reviewText", regexp_replace($"reviewText", "&lt;", "<"))
      .withColumn("reviewText", regexp_replace($"reviewText", "&gt;", ">"))
      .withColumn("reviewText", trim(regexp_replace($"reviewText", "\\s+", " ")))

      // Type Casting for schema consistency
      .withColumn("unixReviewTime", $"unixReviewTime".cast(LongType))
      .withColumn("overall", $"overall".cast(DoubleType))

      // HELPFUL VOTES PARSING: Removing commas and handling nulls
      .withColumn("helpful_votes",
         coalesce(regexp_replace($"vote", ",", "").cast(IntegerType), lit(0))
      )
      .drop("vote", "reviewTime")

      // KEY STANDARDIZATION: Mapping to canonical internal schema
      .withColumnRenamed("reviewerID", "userID")
      .withColumnRenamed("asin", "productID")

    // Persistence in memory to optimize subsequent actions (Count + Extraction)
    cleanedReviewsDF.cache()
    println(s"[INFO] Successfully processed ${cleanedReviewsDF.count()} sampled review records.")

    // ---------------------------------------------------------
    // STEP 2: USER DIMENSION EXTRACTION
    // ---------------------------------------------------------
    println("[INFO] Extracting unique user profiles...")
    val usersDF = cleanedReviewsDF.select("userID", "reviewerName").distinct()

    // ---------------------------------------------------------
    // STEP 3: REVIEW FACT TABLE FINALIZATION
    // ---------------------------------------------------------
    // Removing denormalized names to keep the fact table lean
    val reviewsFinalDF = cleanedReviewsDF.drop("reviewerName")

    // ---------------------------------------------------------
    // STEP 4: PRODUCT METADATA ENRICHMENT
    // ---------------------------------------------------------
    // Filtering Strategy: Extract unique IDs from the sample to reduce join volume
    val uniqueProductIDs = cleanedReviewsDF.select("productID").distinct()

    val metadataInputPath = "data/metadata_electronics.json"
    println(s"[INFO] Loading product metadata from: $metadataInputPath")
    val rawProductsDF = spark.read.json(metadataInputPath)

    // --- PRODUCT DATA TRANSFORMATION ---
    val cleanedProductsDF = rawProductsDF
      .withColumnRenamed("asin", "productID")

      // Price Sanitization: Removing currency symbols and casting to numeric
      .withColumn("price_clean",
         when($"price" === "", null)
         .otherwise(regexp_replace($"price", "\\$", "").cast(DoubleType))
      )
      .drop("price")
      .withColumnRenamed("price_clean", "price")

      // TITLE NORMALIZATION
      .withColumn("title", regexp_replace($"title", "<[^>]+>", " "))
      .withColumn("title", regexp_replace($"title", "&nbsp;", " "))
      .withColumn("title", regexp_replace($"title", "&amp;", "&"))
      .withColumn("title", regexp_replace($"title", "&quot;", "\""))
      .withColumn("title", trim(regexp_replace($"title", "\\s+", " ")))

      // CATEGORY AND TECH FIELD CLEANING
      .withColumn("main_cat", regexp_replace($"main_cat", "&amp;", "&"))
      .withColumn("tech1", regexp_replace($"tech1", "<[^>]+>", ""))
      .withColumn("tech2", regexp_replace($"tech2", "<[^>]+>", ""))

      // RANK CLEANING: Removing non-numeric artifacts
      .withColumn("rank", regexp_replace($"rank", "[\\[\\]\">]", ""))

      // SIMILAR ITEMS: Extracting Amazon Standard Identification Numbers (ASIN)
      .withColumn("similar_item", regexp_extract_all($"similar_item", lit("(B0[0-9A-Z]{8})"), lit(1)))

      // ARRAY TRANSFORMATION: Cleaning descriptions within nested structures
      .withColumn("description", expr("""
         transform(description, x ->
           trim(regexp_replace(
             regexp_replace(
               regexp_replace(x, '<[^>]+>', ' '),
               '&nbsp;', ' '),
             '\\s+', ' ')
           )
         )
      """))

    // --- BROADCAST JOIN OPTIMIZATION ---
    // Using a Broadcast Join to avoid expensive network shuffles.
    // Small distinct ID set is sent to all executors to filter the larger metadata dataset.
    println("[INFO] Executing Broadcast Join for metadata filtering...")
    val filteredProductsDF = cleanedProductsDF.join(broadcast(uniqueProductIDs), Seq("productID"), "inner")

    println(s"[INFO] Successfully filtered metadata for ${filteredProductsDF.count()} unique products.")

    // ---------------------------------------------------------
    // STEP 5: MONGODB DATA PERSISTENCE
    // ---------------------------------------------------------

    // Optimized write configuration for higher throughput
    val mongoWriteOptions = Map(
      "writeConcern.w" -> "1",  // Acknowledge primary write for performance
      "batchSize" -> "2000"     // Increased batch size to minimize network overhead
    )

    println("[INFO] Persisting 'reviews' collection to MongoDB...")
    reviewsFinalDF.write
      .format("mongodb")
      .mode(SaveMode.Overwrite)
      .options(mongoWriteOptions)
      .option("database", "amazon_db")
      .option("collection", "reviews")
      .save()

    println("[INFO] Persisting 'products' collection to MongoDB...")
    filteredProductsDF.write
      .format("mongodb")
      .mode(SaveMode.Overwrite)
      .options(mongoWriteOptions)
      .option("database", "amazon_db")
      .option("collection", "products")
      .save()

    println("[INFO] Persisting 'users' collection to MongoDB...")
    usersDF.write
      .format("mongodb")
      .mode(SaveMode.Overwrite)
      .options(mongoWriteOptions)
      .option("database", "amazon_db")
      .option("collection", "users")
      .save()

    println("[SUCCESS] ETL Pipeline execution completed.")
    spark.stop()
  }
}