import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
 * Product Keyphrase Extraction Module.
 * Identifies distinctive keywords for "Pros" and "Cons" by analyzing
 * high-rated vs. low-rated reviews and filtering out common vocabulary.
 */
object ProductKeyphraseExtraction {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("AmazonProductKeywords")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("[INFO] --- LOADING DATASETS ---")

    val reviewsDF = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .option("database", "amazon_db")
      .option("collection", "reviews")
      .load()
      .select("productID", "reviewText", "overall")
      .na.drop()

    val productsDF = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .option("database", "amazon_db")
      .option("collection", "products")
      .load()
      .select("productID", "title")

    println("[INFO] Identifying top 3 most reviewed products...")

    // Aggregate reviews to find the most popular products
    val topProductIDs = reviewsDF.groupBy("productID")
      .count()
      .sort($"count".desc)
      .limit(3)
      .select("productID")
      .collect()
      .map(_.getString(0))

    println(s"[INFO] Analyzing DISTINCTIVE keywords for the top ${topProductIDs.length} products...\n")

    topProductIDs.foreach { targetProductID =>

      // Retrieve product title
      val productTitle = productsDF.filter($"productID" === targetProductID)
        .select("title")
        .first()
        .getString(0)

      println("===========================================================")
      println(s"PRODUCT: $productTitle (ID: $targetProductID)")

      // Cache the specific reviews for this product to avoid re-fetching
      val productReviewsDF = reviewsDF.filter($"productID" === targetProductID).cache()

      // 1. RAW WORD EXTRACTION
      // We extract top words from both Positive (>3 stars) and Negative (<=3 stars) reviews.
      // We fetch a larger pool (top 50) to ensure we have enough candidates after filtering.
      val rawPosWords = getRawTopWords(spark, productReviewsDF.filter($"overall" > 3), "reviewText", 50)
      val rawNegWords = getRawTopWords(spark, productReviewsDF.filter($"overall" <= 3), "reviewText", 50)

      // 2. INTERSECTION LOGIC
      // We identify words that appear frequently in BOTH groups (e.g., "sound", "speaker", "device").
      // These are neutral context words, not distinctive differentiators.
      val commonWords = rawPosWords.intersect(rawNegWords).toSet

      // 3. SUBTRACTION (DISTINCTIVE FILTERING)
      // We remove the common words to isolate sentiments specific to that sentiment class.
      val distinctPos = rawPosWords.filterNot(commonWords.contains).take(10)
      val distinctNeg = rawNegWords.filterNot(commonWords.contains).take(10)

      println(s"\n[+] PROS (Distinctive Positive Features):")
      if (distinctPos.isEmpty) println("   No distinctive positive features found.")
      else println(s"   ${distinctPos.mkString(", ")}")

      println(s"\n[-] CONS (Distinctive Negative Issues):")
      if (distinctNeg.isEmpty) println("   No distinctive negative issues found.")
      else println(s"   ${distinctNeg.mkString(", ")}")

      println("===========================================================\n")

      // Clear cache for the next iteration
      productReviewsDF.unpersist()
    }

    spark.stop()
  }

  /**
   * Helper method to extract frequent words from a DataFrame.
   * Performs cleaning, tokenization, stop-word removal, and frequency counting.
   */
  def getRawTopWords(spark: SparkSession, df: DataFrame, colName: String, limit: Int): Array[String] = {
    import spark.implicits._

    if (df.count() == 0) return Array.empty[String]

    // Remove non-alphabetic characters
    val cleanedDF = df.withColumn("clean_text", regexp_replace(col(colName), "[^a-zA-Z ]", "").cast("string"))

    val tokenizer = new Tokenizer()
      .setInputCol("clean_text")
      .setOutputCol("words")

    val wordsDF = tokenizer.transform(cleanedDF)

    // Standard English stop words removal
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val filteredDF = stopWordsRemover.transform(wordsDF)

    // Custom exclusion list for generic e-commerce terms that add no semantic value
    val ecommerceStopWords = Seq(
      "product", "one", "great", "good", "bad", "use", "get", "like", "just", "amazon", "bought",
      "will", "time", "work", "works", "don't", "does", "didn't", "thing", "purchase", "review",
      "really", "much", "well", "even", "back", "make"
    )

    // Explode arrays into individual rows, filter, count, and sort
    filteredDF
      .select(explode($"filtered_words").as("word"))
      .filter(length($"word") > 3) // Skip short words
      .filter(!($"word".isin(ecommerceStopWords: _*)))
      .groupBy("word")
      .count()
      .sort($"count".desc)
      .limit(limit)
      .select("word")
      .as[String]
      .collect()
  }
}