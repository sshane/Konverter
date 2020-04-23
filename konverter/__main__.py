from tensorflow import keras
from konverter import Konverter
import typer


def main(input_model:str, output_script:str):
    model = keras.models.load_model(input_model)
    konverter = Konverter(model, output_file=output_script, indent_spaces=2)


def run():
    typer.run(main)


if __name__ == '__main__':
    run()
