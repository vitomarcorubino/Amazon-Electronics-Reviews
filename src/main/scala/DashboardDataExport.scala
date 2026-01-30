import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Dashboard Data Export Module.
 * Aggregates key performance indicators (KPIs) for Brands, Categories, and Users
 * and exports them as CSV files for dashboard visualization.
 */
object DashboardDataExport {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("AmazonDashboardExport")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("[INFO] --- STARTING DASHBOARD DATA EXPORT ---")

    // ---------------------------------------------------------
    // STEP 1: DATA LOADING
    // ---------------------------------------------------------
    // We select only the necessary columns to minimize memory usage during the join.
    val reviewsDF = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.reviews")
      .option("database", "amazon_db")
      .option("collection", "reviews")
      .load()
      .select("productID", "userID", "overall", "helpful_votes")

    val productsDF = spark.read.format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .option("database", "amazon_db")
      .option("collection", "products")
      .load()
      .select("productID", "brand", "main_cat", "price")

    // OPTIMIZATION: Perform the join once and cache the result for multiple aggregations.
    val enrichedReviewsDF = reviewsDF.join(productsDF, "productID").cache()

    // ---------------------------------------------------------
    // KPI 1: TOP BRANDS BY ESTIMATED REVENUE
    // ---------------------------------------------------------
    val brandRevenueDF = enrichedReviewsDF
      .groupBy("brand")
      .agg(
        count("productID").as("sales_volume"),
        round(avg("price"), 2).as("avg_price"),
        round(sum("price"), 2).as("estimated_revenue"),
        round(avg("overall"), 2).as("avg_rating")
      )
      .filter($"brand".isNotNull && $"brand" =!= "") // Exclude missing brand names
      .sort($"estimated_revenue".desc)
      .limit(50)

    println("[INFO] Exporting Top Brands Report...")
    brandRevenueDF.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv("output_dashboard/top_brands")

    // ---------------------------------------------------------
    // KPI 2: CATEGORY PERFORMANCE & NEGATIVE SENTIMENT
    // ---------------------------------------------------------
    val categoryPerformanceDF = enrichedReviewsDF
      .groupBy("main_cat")
      .agg(
        count("overall").as("total_reviews"),
        round(avg("overall"), 2).as("avg_rating"),
        // Calculate the ratio of negative reviews (< 3 stars) to identify problem areas
        round(sum(when($"overall" < 3, 1).otherwise(0)) / count("overall"), 4).as("negative_ratio")
      )
      .filter($"total_reviews" > 100) // Filter for statistically significant categories
      .sort($"negative_ratio".desc)

    println("[INFO] Exporting Category Performance Report...")
    categoryPerformanceDF.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv("output_dashboard/category_issues")

    // ---------------------------------------------------------
    // KPI 3: TOP REVIEWERS (INFLUENCERS)
    // ---------------------------------------------------------
    // We use the original reviewsDF here as product metadata is not required.
    // This reduces overhead.
    val topUsersDF = reviewsDF
      .groupBy("userID")
      .agg(
        count("productID").as("reviews_count"),
        sum("helpful_votes").as("total_helpful_votes"),
        round(avg("overall"), 2).as("avg_score")
      )
      .sort($"total_helpful_votes".desc)
      .limit(100)

    println("[INFO] Exporting Top Users Report...")
    topUsersDF.coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv("output_dashboard/top_users")

    // Unpersist cached data
    enrichedReviewsDF.unpersist()

    spark.stop()
    println("[SUCCESS] Export completed. Please check the 'output_dashboard' directory.")
  }
}