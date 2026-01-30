ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.1",
  "org.apache.spark" %% "spark-sql"  % "3.5.1",
  "org.apache.spark" %% "spark-mllib"% "3.5.1",
  "org.apache.spark" %% "spark-streaming"% "3.5.1",
  "org.apache.spark" %% "spark-graphx"% "3.5.1",

  "org.mongodb.spark" %% "mongo-spark-connector" % "10.3.0"
)

lazy val root = (project in file("."))
  .settings(
    name := "Amazon_Electronics_Reviews"
  )
