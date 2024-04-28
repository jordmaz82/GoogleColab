from tflite_model_maker import searcher
import os

def pretty_print(statement):
    print("*******************************************", end='/n/n')
    print(statement, end='/n/n')
    print("*******************************************", end='/n/n')


data_loader = searcher.TextDataLoader.create("universal_sentence_encoder.tflite", l2_normalize=True)
data_loader.load_from_csv("cnn_dailymail.csv", text_column="highlights", metadata_column="urls")
scann_options = searcher.ScaNNOptions(
      distance_measure="dot_product",
      tree=searcher.Tree(num_leaves=140, num_leaves_to_search=4),
      score_ah=searcher.ScoreAH(dimensions_per_block=1, anisotropic_quantization_threshold=0.2))
model = searcher.Searcher.create_from_data(data_loader, scann_options)

pretty_print("Exporting Model....")
try:
    model.export(
      export_filename="searcher.tflite",
      userinfo="",
      export_format=searcher.ExportFormat.TFLITE)

    if os.path.exists("searcher.tflite"):
        pretty_print("searcher.tflite file was created")
    else:
        pretty_print("searcher.tflite file was not created")
except Exception as e:
    pretty_print("Error exporting model: %s"%e)


