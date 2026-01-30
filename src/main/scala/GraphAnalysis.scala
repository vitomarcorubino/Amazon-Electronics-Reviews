import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.graphx.lib.LabelPropagation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

/**
 * Graph Analysis Module using Spark GraphX.
 * Analyzes product relationships based on "also_buy" behavior to identify
 * influential products (PageRank) and product communities (Label Propagation).
 */
object GraphAnalysis {
  def main(args: Array[String]): Unit = {

    // 1. Spark Session Setup
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("AmazonGraphAnalysis")
      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("[INFO] --- STARTING GRAPH ANALYSIS PIPELINE ---")

    // ---------------------------------------------------------
    // STEP 1: DATA INGESTION
    // ---------------------------------------------------------
    // We load product metadata, specifically focusing on the 'also_buy' array
    // which defines the edges of our graph.
    val productsDF = spark.read
      .format("mongodb")
      .option("uri", "mongodb://127.0.0.1:27017/amazon_db.products")
      .option("database", "amazon_db")
      .option("collection", "products")
      .load()
      .select("productID", "also_buy", "title")
      .na.drop(Seq("productID"))

    // ---------------------------------------------------------
    // STEP 2: VERTEX GENERATION (NODES)
    // ---------------------------------------------------------
    // GraphX requires vertices to have unique Long IDs.
    // Since our productIDs are Strings (ASINs), we map them to Long using hashCode.
    // Note: Hash collisions are possible but acceptable for this analytical demonstration.
    val verticesRDD: RDD[(Long, String)] = productsDF
      .select("productID", "title")
      .rdd
      .map(row => {
        val idString = row.getString(0)
        val title = if (row.isNullAt(1)) "No Title" else row.getString(1)
        (idString.hashCode.toLong, title)
      })

    // ---------------------------------------------------------
    // STEP 3: EDGE GENERATION (RELATIONSHIPS)
    // ---------------------------------------------------------
    // We explode the 'also_buy' array to create directed edges: Source Product -> Target Product
    val distinctEdgesDF = productsDF
      .withColumn("target_product", explode($"also_buy"))
      .select("productID", "target_product")
      .na.drop()

    val edgesRDD: RDD[Edge[Int]] = distinctEdgesDF.rdd.map(row => {
      val sourceId = row.getString(0).hashCode.toLong
      val targetId = row.getString(1).hashCode.toLong
      Edge(sourceId, targetId, 1) // Edge weight set to 1
    })

    println(s"[INFO] Graph structure constructed: ${verticesRDD.count()} vertices and ${edgesRDD.count()} edges.")

    // ---------------------------------------------------------
    // STEP 4: GRAPH CONSTRUCTION
    // ---------------------------------------------------------
    val productGraph = Graph(verticesRDD, edgesRDD)

    // ---------------------------------------------------------
    // STEP 5: PAGERANK ALGORITHM
    // ---------------------------------------------------------
    println("[INFO] Executing PageRank algorithm (Tolerance: 0.001)...")

    // Calculate importance of each node based on incoming links.
    val pageRankGraph = productGraph.pageRank(0.001)

    // Extract top 10 most influential products by joining ranks with titles
    val topInfluentialProducts = pageRankGraph.vertices.join(verticesRDD)
      .map { case (id, (rank, title)) => (rank, title, id) }
      .sortBy(_._1, ascending = false)
      .take(10)

    println("\n=== TOP 10 MOST INFLUENTIAL PRODUCTS (PAGERANK) ===")
    topInfluentialProducts.foreach { case (rank, title, id) =>
      println(f"Rank: $rank%2.4f | Product: $title")
    }

    // ---------------------------------------------------------
    // STEP 6: CONNECTED COMPONENTS
    // ---------------------------------------------------------
    // Identifying isolated sub-graphs (clusters) within the network.
    val connectedComponentsGraph = productGraph.connectedComponents()

    val largestClusterSize = connectedComponentsGraph.vertices
      .map { case (_, clusterId) => (clusterId, 1) }
      .reduceByKey(_ + _)
      .map(_._2)
      .max()

    println(s"\n[RESULT] The largest connected cluster contains $largestClusterSize interconnected products.")

    // ---------------------------------------------------------
    // STEP 7: COMMUNITY DETECTION (LABEL PROPAGATION)
    // ---------------------------------------------------------
    println("\n[INFO] Running Label Propagation for community detection...")

    // Label Propagation finds dense communities by propagating labels through the graph.
    val communitiesGraph = LabelPropagation.run(productGraph, 5) // 5 iterations

    // Join community IDs back with Product Titles for readability
    val communityAnalysisRDD = communitiesGraph.vertices.join(verticesRDD)
      .map { case (id, (communityId, title)) => (communityId, title) }

    // Identify the 3 largest communities
    val topCommunityIDs = communityAnalysisRDD
      .map { case (cid, title) => (cid, 1) }
      .reduceByKey(_ + _)
      .sortBy(_._2, ascending = false)
      .take(3)
      .map(_._1)

    // Display sample products from these top communities
    topCommunityIDs.foreach(cid => {
      println(f"\n--- Community ID: $cid ---")
      communityAnalysisRDD.filter(_._1 == cid).take(10).foreach(x => println(s" - ${x._2}"))
    })

    spark.stop()
  }
}