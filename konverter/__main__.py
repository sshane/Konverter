import sys
import argparse
import konverter
from konverter.utils.general import success, info, warning, error, COLORS, color_logo, blue_grad

KONVERTER_VERSION = "v0.2.4.1"  # fixme: unify this
KONVERTER_LOGO_COLORED = color_logo(KONVERTER_VERSION)


class KonverterCLI:
  def __init__(self, args):
    self.args = args

    self._name = 'konverter'
    self._arguments = {'input_model': {'help': 'The path of your tf.keras .h5 model', 'required': True},
                       'output_file': {'help': 'Optional file path for your output model, along with the weights file. Default is same name, same dir', 'required': False}}
    self._flags = {'--indent': {'alias': '-i', 'help': 'How many spaces to use for indentation, default is 2', 'type': int, 'default': 2},
                   '--silent': {'alias': '-s', 'help': 'Whether you want Konverter to silently Konvert', 'type': bool, 'default': False},
                   '--no-watermark': {'alias': '-nw', 'help': 'Removes the watermark prepended to the output model file', 'type': bool}}
    self.parser = argparse.ArgumentParser()
    self._setup_parser()
    self._sanitize_args()
    self._konvert()

  def _konvert(self):
    if len(self.args) == 0:
      self._print_usage(logo=True)
      return
    else:
      flags, e = self.parse_flags()
      if e is not None:
        error(e)
        return

      konverter.konvert(flags.input_model, output_file=flags.output_file, indent=flags.indent, silent=flags.silent, no_watermark=flags.no_watermark)

  def _sanitize_args(self):
    self.args = [arg.strip() for arg in self.args if arg.strip() != '']

  def _print_usage(self, logo=False):
    if logo:
      print(KONVERTER_LOGO_COLORED, end='\n\n')
    warning('A tool to konvert your Keras models into pure Python 🐍+ NumPy.')
    arguments = ' '.join(['{}'.format(arg) if arg_info['required'] else '[{}]'.format(arg) for arg, arg_info in self._arguments.items()])
    flags = info(' [--flags]', ret=True)
    k_color_idxs = [2, 3, 4, 5, 5, 4, 4, 3, 2]
    konverter_colored = ''.join([COLORS.BASE.format(blue_grad[idx]) + c for c, idx in zip(self._name, k_color_idxs)])
    print(COLORS.BASE.format(219) + 'Usage: {} '.format(konverter_colored) + COLORS.BASE.format(85) + arguments + flags + COLORS.ENDC, end='\n\n')

    print(COLORS.BASE.format(85) + 'Arguments 💢:' + COLORS.ENDC)
    for arg, arg_info in self._arguments.items():
      arg = '{}'.format(arg).format(arg)
      print(COLORS.BASE.format(85) + '  {}:{} {}'.format(arg, COLORS.PROMPT, arg_info['help']))
    print()

    info('Flags 🎌:')
    for flag, flag_info in self._flags.items():
      help = flag_info['help']
      if flag_info['alias'] == '':
        pass
      else:
        flag = ', '.join([flag, flag_info['alias']])
      info('  {}:{} {}'.format(flag, COLORS.PROMPT, help))

  def _setup_parser(self):
    for arg, arg_info in self._arguments.items():
      nargs = None
      if not arg_info['required']:
        nargs = '?'
      self.parser.add_argument(arg, help=arg_info['help'], action='store', type=str, nargs=nargs)

    for flag, flag_info in self._flags.items():
      flag_args = {}
      if flag_info['alias'] != '':
        flag = [flag, flag_info['alias']]
      else:
        flag = [flag]
      if flag_info['type'] == bool:
        flag_args['action'] = 'store_true'
      else:
        flag_args['action'] = 'store'
        flag_args['type'] = flag_info['type']
      if 'default' in flag_info:
        flag_args['default'] = flag_info['default']

      self.parser.add_argument(*flag, help=flag_info['help'], **flag_args)

  def parse_flags(self):
    try:
      return self.parser.parse_args(self.args), None
    except Exception as e:
      return None, e


def run():
  args = sys.argv[1:]
  KonverterCLI(args)


if __name__ == '__main__':
  run()
