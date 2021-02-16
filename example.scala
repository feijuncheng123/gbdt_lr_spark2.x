package example

import org.apache.spark.ml.gbtlr.GBTLR
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{GBTClassificationModel,LogisticRegression}

object example {
	def main(args: Array[String]): Unit = {
		val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

		val gbdtModel=GBTClassificationModel.load("gbdt_model_path")

		val gbdtlr=new GBTLR(gbdtModel)
		val predictLeafUDF = udf{features: Vector =>GBTLR.predictLeaf(features)}

    val sampleData=spark.read.parquet("sampleData_path")
		val gbdt_leaf_features = sampleData.withColumn("newFeatures",predictLeafUDF($"features"))

    val lr=new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("newFeatures")
      .setRegParam(0.1)
      .setMaxIter(20)
      .setStandardization(false)
      .setAggregationDepth(10)
    val lrModel=lr.fit(gbdt_leaf_features)

   spark.stop()
	}
}