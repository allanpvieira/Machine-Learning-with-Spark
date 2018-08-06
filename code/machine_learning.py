import findspark
findspark.init()

from pyspark.sql import SQLContext
import pyspark

sc = pyspark.SparkContext(appName="Calories")
sqlContext = SQLContext(sc)
fs_source = "/home/up/Documents/databases/"
fs_output = "/home/up/Documents/output/"

creditcard_df = (sqlContext.read.format("com.databricks.spark.csv").options(header=True, inferSchema=True, delimiter=",")
                 .load(fs_source+"ehresp_2014.csv"))
				 
creditcard_df.show(1)

creditcard_df.printSchema()

#Removendo colunas que não serão inicialmente usadas
temp_df = (creditcard_df.drop("tucaseid").drop("tulineno")
           .drop("erhhch").drop("eeincome1").drop("exincome1").drop("euincome2").drop("erspemch").drop("ethgt")
            .drop("etwgt").drop("euffyday").drop("eufdsit").drop("eusnap").drop("eugenhth").drop("eugroshp")
            .drop("euhgt").drop("euinclvl").drop("euprpmel").drop("eustores").drop("eustreason")
            .drop("eutherm").drop("euwgt").drop("euwic").drop("ertpreat").drop("eufinlwgt")
            .drop("erincome"))
			
temp_df.printSchema()

temp_df.select("*").show(1)

from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import IntegerType

temp_df = temp_df.where("erbmi>=1")


def to_zero(x):
    if x < 0 :
        return -1
    else:
        return x

udf = UserDefinedFunction(lambda x: to_zero(x), IntegerType())

temp_df = temp_df.withColumn('ertseat1', udf(temp_df.ertseat))
temp_df = temp_df.withColumn('eudietsoda1', udf(temp_df.eudietsoda))
temp_df = temp_df.withColumn('eudrink1', udf(temp_df.eudrink))
temp_df = temp_df.withColumn('eueat1', udf(temp_df.eueat))
temp_df = temp_df.withColumn('euexercise1', udf(temp_df.euexercise))
temp_df = temp_df.withColumn('euexfreq1', udf(temp_df.euexfreq))
temp_df = temp_df.withColumn('eufastfdfrq1', udf(temp_df.eufastfdfrq))
temp_df = temp_df.withColumn('eumeat1', udf(temp_df.eumeat))
temp_df = temp_df.withColumn('eumilk1', udf(temp_df.eumilk))
temp_df = temp_df.withColumn('eusoda1', udf(temp_df.eusoda))
temp_df = temp_df.withColumn('eufastfd1', udf(temp_df.eufastfd))

temp_df = (temp_df
           .drop('ertseat')
            .drop('eudietsoda')
            .drop('eudrink')
            .drop('eueat')
            .drop('euexercise')
            .drop('euexfreq')
            .drop('eufastfdfrq')
            .drop('eumeat')
            .drop('eumilk')
            .drop('eusoda')
           .drop('eufastfd'))
		   
		   
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import FloatType

temp_df = temp_df.where("erbmi>=1")

def modify_values(erbmi):
    if erbmi >= 25:
        return 1.0
    else:
        return 0.0

udf = UserDefinedFunction(lambda x: modify_values(x), FloatType())

temp_df = temp_df.withColumn('target_column', udf(temp_df.erbmi))
temp_df=temp_df.drop("erbmi")

# Dividindo os dados em treinamento e teste
weights = [0.9, 0.1]
seed = 13579

training_df, test_df = temp_df.randomSplit(weights, seed)
training_df.cache()
test_df.cache()

print("Número de linhas do Dataframe de treinamento: {0}".format(training_df.count()))
print("Número de linhas do Dataframe de teste: {0}".format(test_df.count()))
print("Distribuicao de fraudes no Dataframe de treinamento: {0}".format(training_df.groupBy("target_column").count().take(4)))

# Efetuando as transformações para gerar features e preparar dados para passagem ao indutor
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

features = training_df.columns[0:10]

feature_assembler = VectorAssembler(inputCols=features, outputCol="features")
label_indexer = StringIndexer(inputCol="target_column", outputCol="label")
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=3, 
                            featureSubsetStrategy="auto", impurity="gini", maxDepth=4, maxBins=32, seed=seed)

instance = [feature_assembler, label_indexer, rf]
pipeline = Pipeline(stages=instance)

model_df = pipeline.fit(training_df)
transformed = model_df.transform(test_df)

#transformed.select("erbmi", "prediction").show(50)
transformed.printSchema()

transformed.select("prediction","label").orderBy("prediction", "label").show(10000);

# Gerando métricas dos dados [TIRAR]
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

results_rfc = trasformed_new.select(['target_string', 'label_string'])
results_collect = results_rfc.collect()

results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = metric(scoreAndLabels)
print("Área sob a curva de Precision/Recall (PR): {0:.2f}".format(metrics.areaUnderPR * 100))
print("O ROC score para 3 arvores: {0:.2f}".format(metrics.areaUnderROC * 100))


# Gerando métricas adicionais
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predAndLabel_mlp = transformed.select(['prediction', 'label'])

accuracy_rfc = MulticlassClassificationEvaluator(metricName="accuracy")
f1_rfc = MulticlassClassificationEvaluator(metricName="f1")

print ("Accuracy MLP: {0}".format(accuracy_rfc.evaluate(predAndLabel_mlp)))
print ("F1-Score MLP: {0}".format(f1_rfc.evaluate(predAndLabel_mlp)))