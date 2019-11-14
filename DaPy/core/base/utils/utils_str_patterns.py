from re import compile as re_compile

PATTERN_AND_OR = re_compile(r'\sand\s|\sor\s')
PATTERN_COMBINE = re_compile(r'[(](.*?)(\sand\s|\sor\s)(.*?)[)]')
PATTERN_RECOMBINE = re_compile(r'[)](\sand\s|\sor\s)[(]')
PATTERN_COLUMN = r'[(|\s]{0,1}%s[\s|)]{0,1}'

PATTERN_EQUAL = re_compile(r'(.*?)(!=|==)(.*?)')
PATTERN_LESS = re_compile(r'(.*?)(<=|<)(.*?)')
PATTERN_GREAT = re_compile(r'(.*?)(>=|>)(.*?)')
PATTERN_BETWEEN1 = re_compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')
PATTERN_BETWEEN2 = re_compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')
PATTERN_EQUALS = (PATTERN_EQUAL, PATTERN_LESS, PATTERN_GREAT)
SIMPLE_EQUAL_PATTERN = ('!=', '<=', '>=')

PATTERN_CHANGE_LINE = re_compile(r'[\n|\r]')
