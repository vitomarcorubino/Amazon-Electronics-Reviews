import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._

object SentimentAnalysis {
  def main(args: Array[String]): Unit = {

    // 1. Setup Spark Session
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("SentimentAnalysis")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("===============================================================")
    println("   AMAZON UNIFIED NLP: FROM GLOBAL SENTIMENT TO PRODUCT DETAILS")
    println("===============================================================")

    // 2. Load Data & Clean
    println("--- [PHASE 1] LOADING DATA ---")

    val reviews = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .option("database", "amazon_db").option("collection", "reviews")
      .load()
      .select("reviewText", "overall", "productID")
      .na.drop()
      .filter(length($"reviewText") > 5)

    // Cleaning text (Regex): we keep only letters and spaces, lowercase everything
    val cleanReviews = reviews.withColumn("text_cleaned",
      trim(regexp_replace(lower($"reviewText"), "[^a-z ]", " "))
    )

    cleanReviews.cache()
    println(s"Total reviews loaded and cleaned: ${cleanReviews.count()}")

    val products = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .option("database", "amazon_db").option("collection", "products")
      .load()
      .select("productID", "brand", "title")

    // ==============================================================================
    // PHASE 2: GLOBAL SENTIMENT MODEL
    // ==============================================================================
    println("\n--- [PHASE 2] GLOBAL SENTIMENT MODELLING ---")

    val data = cleanReviews.withColumn("label", when($"overall" > 3.0, 1.0).otherwise(0.0))
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Pipeline NLP
    val tokenizer = new Tokenizer().setInputCol("text_cleaned").setOutputCol("words")

    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val globalStopWords = StopWordsRemover.loadDefaultStopWords("english") ++ Array("product", "amazon", "bought",
      "item", "review", "purchased", "one")
    remover.setStopWords(globalStopWords)

    val cv = new CountVectorizer().setInputCol("filtered_words").setOutputCol("rawFeatures").setVocabSize(10000).setMinDF(10.0)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, cv, idf, lr))

    println("Training Global Model...")
    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)
    val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    println(s"Model Accuracy (AUC): ${evaluator.evaluate(predictions)}")

    // Extract Global Keywords
    println("\n>>> GLOBAL MARKET DRIVERS (What defines 'Good' vs 'Bad'):")
    val cvModel = model.stages(2).asInstanceOf[CountVectorizerModel]
    val lrModel = model.stages(4).asInstanceOf[LogisticRegressionModel]
    val wordWeights = cvModel.vocabulary.zip(lrModel.coefficients.toArray)

    println("Top 10 Positive Words:")
    wordWeights.sortBy(-_._2).take(10).foreach { case (w, s) => println(f"  + $w%-15s ($s%2.2f)") }

    println("Top 10 Negative Words:")
    wordWeights.sortBy(_._2).take(10).foreach { case (w, s) => println(f"  - $w%-15s ($s%2.2f)") }

    // ==============================================================================
    // PHASE 3: BRAND INTELLIGENCE
    // ==============================================================================
    println("\n--- [PHASE 3] BRAND INTELLIGENCE ---")

    val richPredictions = predictions.join(products, "productID")
    richPredictions.createOrReplaceTempView("sentiment_data")

    val topBrands = spark.sql("""
      SELECT brand, count(*) as reviews, round(avg(overall), 2) as rating,
             round(avg(prediction) * 100, 1) as sentiment_score
      FROM sentiment_data
      WHERE brand IS NOT NULL AND brand != ''
      GROUP BY brand HAVING reviews > 20
      ORDER BY rating DESC LIMIT 5
    """)
    topBrands.show(false)

    // ==============================================================================
    // PHASE 4: PRODUCT DEEP DIVE (Feature Extraction - INLINE)
    // ==============================================================================
    println("\n--- [PHASE 4] PRODUCT DEEP DIVE (DIFFERENTIAL ANALYSIS) ---")

    val topProducts = cleanReviews.groupBy("productID").count().sort($"count".desc).limit(3)
      .select("productID").collect().map(_.getString(0))

    // Prepariamo gli strumenti NLP per l'uso inline dentro il loop
    val keywordTokenizer = new Tokenizer().setInputCol("text_cleaned").setOutputCol("words")
    val keywordRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

    // Forbidden words list to avoid false positives
    val banList = Seq("product", "one", "great", "good", "bad", "use", "get", "like", "just", "amazon", "bought",
      "will", "time", "work", "works", "don't", "does", "didn't", "thing", "purchase", "review", "really", "much",
      "well", "even", "back", "make", "price", "would")
    keywordRemover.setStopWords(StopWordsRemover.loadDefaultStopWords("english") ++ banList)

    topProducts.foreach { targetProductID =>
      val titleRow = products.filter($"productID" === targetProductID).select("title").head()
      val title = if (titleRow != null) titleRow.getString(0) else "Unknown Product"

      println(s"\n--- Analyzing: ${title.take(60)}... (ID: $targetProductID) ---")

      val productReviews = cleanReviews.filter($"productID" === targetProductID).cache()

      // --- Inline positive analysis ---
      val posReviews = productReviews.filter($"overall" > 3)
      val rawPosWords = if (posReviews.count() > 0) {
         keywordRemover.transform(keywordTokenizer.transform(posReviews))
          .select(explode($"filtered").as("word"))
          .filter(length($"word") > 3)
          .groupBy("word").count().sort($"count".desc)
          .limit(50).select("word").as[String].collect()
      } else Array.empty[String]

      // --- Inline negative analysis ---
      val negReviews = productReviews.filter($"overall" <= 3)
      val rawNegWords = if (negReviews.count() > 0) {
         keywordRemover.transform(keywordTokenizer.transform(negReviews))
          .select(explode($"filtered").as("word"))
          .filter(length($"word") > 3)
          .groupBy("word").count().sort($"count".desc)
          .limit(50).select("word").as[String].collect()
      } else Array.empty[String]

      // Negative and Positive words intersection
      val commonWords = rawPosWords.intersect(rawNegWords).toSet
      val distinctPos = rawPosWords.filterNot(commonWords.contains).take(8)
      val distinctNeg = rawNegWords.filterNot(commonWords.contains).take(8)

      println(s"[+] PROS: ${distinctPos.mkString(", ")}")

      if (distinctNeg.isEmpty) println("[-] CONS: None (Perfect Product?)")
      else println(s"[-] CONS: ${distinctNeg.mkString(", ")}")

      productReviews.unpersist()
    }

    spark.stop()
  }
}