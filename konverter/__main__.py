from konverter import Konverter
import typer

app = typer.Typer()


@app.command()
def main(input_model: str, output_model: str, indentation: int = 2, verbose: bool = True, use_watermark: bool = True):
  """
  Generates a Konverted model from the input Keras model

  str:input_model: The the location of your tf.keras .h5 model
  str:output_model: The location you would like your model saved to, along with the weights file

  Optional:
  bool:verbose: Whether you want Konverter to print its current status in Konversion
  bool:use_watermark: Whether you want a watermark prepended to the output model file
  """
  Konverter(input_model, output_file=output_model, indent_spaces=indentation, verbose=verbose, use_watermark=use_watermark)


if __name__ == '__main__':
  app()
