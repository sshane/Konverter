from tensorflow import keras
from konverter import Konverter
import typer


def main(input_model: str, output_script: str, indentation: int = 2, verbose: bool = True, use_watermark: bool = True):
  model = keras.models.load_model(input_model)
  konverter = Konverter(model, output_file=output_script, indent_spaces=indentation,
                        verbose=verbose, use_watermark=use_watermark)


def run():
  typer.run(main)


if __name__ == '__main__':
  run()
