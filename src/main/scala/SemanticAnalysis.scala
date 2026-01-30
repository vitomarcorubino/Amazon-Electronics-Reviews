import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Semantic Analysis Module using Word2Vec.
 * Implements a "Load-or-Train" mechanism to handle vector space models for review text.
 */
object SemanticAnalysis {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("AmazonSemanticAnalysis")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val modelOutputPath = "models/word2vec_electronics_v1"

    // ---------------------------------------------------------
    // LOAD-OR-TRAIN LOGIC
    // ---------------------------------------------------------
    // The try-catch block determines the model source.
    // If a saved model exists, it is loaded; otherwise, a new one is trained.
    val word2VecModel = try {
      println(s"[INFO] Checking for existing model at: $modelOutputPath")
      val existingModel = Word2VecModel.load(modelOutputPath)
      println("[SUCCESS] Model loaded from disk. Skipping training phase.")
      existingModel
    } catch {
      case _: Exception =>
        println("[INFO] Model not found locally. Initiating training pipeline...")

        println("[INFO] Loading review data from MongoDB...")
        val rawReviewsDF = spark.read.format("mongodb")
          .option("uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
          .option("database", "amazon_db")
          .option("collection", "reviews")
          .load()
          .select("reviewText")
          .na.drop()
          .filter(length($"reviewText") > 10)

        println(s"[INFO] Training dataset size: ${rawReviewsDF.count()} reviews")

        // --- NLP PREPROCESSING PIPELINE ---
        println("[INFO] Preprocessing text data...")
        // Remove non-alphabetic characters and convert to lowercase
        val cleanedReviewsDF = rawReviewsDF.withColumn("clean_text",
          regexp_replace(lower(col("reviewText")), "[^a-z ]", " ")
        )

        // Tokenize text into individual words
        val tokenizer = new Tokenizer()
          .setInputCol("clean_text")
          .setOutputCol("words")

        // Remove common English stop words
        val stopWordsRemover = new StopWordsRemover()
          .setInputCol("words")
          .setOutputCol("filtered_words")

        val pipelineDataDF = stopWordsRemover.transform(tokenizer.transform(cleanedReviewsDF))

        // --- MODEL CONFIGURATION & TRAINING ---
        val word2VecEstimator = new Word2Vec()
          .setInputCol("filtered_words")
          .setOutputCol("result_vector")
          .setVectorSize(100)
          .setMinCount(50)
          .setWindowSize(5)

        println("[INFO] Learning vector space (this may take some time)...")
        val trainedModel = word2VecEstimator.fit(pipelineDataDF)

        println(s"[INFO] Persisting trained model to: $modelOutputPath")
        trainedModel.write.overwrite().save(modelOutputPath)

        // Return the newly trained model to the variable
        trainedModel
    }

    // ---------------------------------------------------------
    // SEMANTIC QUERYING
    // ---------------------------------------------------------
    println("\n--- SEMANTIC SEARCH (SYNONYM DISCOVERY) ---")
    val queryTerms = Seq("battery", "screen", "price", "broken", "sound", "amazon", "quality", "expensive")

    queryTerms.foreach { term =>
      try {
        println(s"\n[RESULT] Concepts semantically associated with '$term':")
        val synonymsDF = word2VecModel.findSynonyms(term, 10)

        synonymsDF.collect().foreach { row =>
          val synonym = row.getString(0)
          val cosineSimilarity = row.getDouble(1)

          // Filter out the word itself if it appears in the results
          if (synonym != term) {
            println(f"   $synonym%-15s (Similarity: $cosineSimilarity%1.4f)")
          }
        }
      } catch {
        case _: Exception => println(s"   [WARN] Word '$term' was not found in the vocabulary.")
      }
    }

    spark.stop()
  }
}