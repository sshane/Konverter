def error(msg, end='\n', ret=False):
  """
  The following applies to error, warning, and success methods
  :param msg: The message to display
  :param end: The ending char, default is \n
  :param ret: Whether to return the formatted string, or print it
  :return: The formatted string if ret is True
  """
  e = '{}{}{}'.format(COLORS.FAIL, msg, COLORS.ENDC)
  if ret:
    return e
  print(e, end=end)


def warning(msg, end='\n', ret=False):
  w = '{}{}{}'.format(COLORS.PROMPT, msg, COLORS.ENDC)
  if ret:
    return w
  print(w, end=end)


def success(msg, end='\n', ret=False):
  s = '{}{}{}'.format(COLORS.SUCCESS, msg, COLORS.ENDC)
  if ret:
    return s
  print(s, end=end)


def info(msg, end='\n', ret=False):
  s = '{}{}{}'.format(COLORS.PRETTY_YELLOW, msg, COLORS.ENDC)
  if ret:
    return s
  print(s, end=end)


class COLORS:
  BASE = '\33[38;5;{}m'  # seems to support more colors
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  CBLUE = '\33[44m'
  BOLD = '\033[1m'
  OKGREEN = '\033[92m'
  CWHITE = '\33[37m'
  ENDC = '\033[0m' + CWHITE
  UNDERLINE = '\033[4m'
  PINK = '\33[38;5;207m'
  PRETTY_YELLOW = BASE.format(220)

  RED = '\033[91m'
  PURPLE_BG = '\33[45m'
  YELLOW = '\033[93m'
  BLUE_GREEN = BASE.format(85)

  FAIL = RED
  INFO = PURPLE_BG
  SUCCESS = OKGREEN
  PROMPT = YELLOW
  CYAN = '\033[36m'
  WARNING = '\033[33m'


KONVERTER_LOGO = " _   __                          _\n" \
                 "| | / /                         | |\n" \
                 "| |/ /  ___  _ ____   _____ _ __| |_ ___ _ __ {}\n" \
                 "|    \\ / _ \\| '_ \\ \\ / / _ \\ '__| __/ _ \\ '__|\n" \
                 "| |\\  \\ (_) | | | \\ V /  __/ |  | ||  __/ |\n" \
                 "\\_| \\_/\\___/|_| |_|\\_/ \\___|_|   \\__\\___|_|"



blue_grad = [27, 33, 39, 45, 51, 87]
def color_logo(version):
  KONVERTER_LOGO_V = KONVERTER_LOGO.format(COLORS.BASE.format(212) + version + '!')
  lines = KONVERTER_LOGO_V.split('\n')
  lines = [COLORS.BASE.format(col) + line + COLORS.ENDC for line, col in zip(lines, blue_grad)]
  return '\n'.join(lines)
