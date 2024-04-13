import re

from typing import List

STYLE_TO_CHAR = {
    "dash": "-",
    "star": "*",
}
NUM_BULLET_PATTERM = re.compile(r"^\d+(\.)?\s(.*?)$")
GEN_BULLET_PATTERN = re.compile(r"^[*-]\s(.*?)$")


def string_to_list(xs: str, style: str = "dash") -> List[str]:
    """ Convert a string with embedded newlines to a list of strings
        separated by newlines.

        :param xs: string with embedded newlines
        :param style: bullet style to strip (dash, star, number)
        :return: a list of strings
    """
    stripped_xs = []
    for x in xs.split("\n"):
        if NUM_BULLET_PATTERM.match(x):
            stripped_x = NUM_BULLET_PATTERM.match(x).group(2)
        elif GEN_BULLET_PATTERN.match(x):
            stripped_x = GEN_BULLET_PATTERN.match(x).group(1)
        else:
            stripped_x = x
        stripped_xs.append(stripped_x)
    return stripped_xs


def list_to_string(xs: List[str], style: str = "dash") -> str:
    """ Convert a list of strings to a string with embedded newlines.
        Strips (known) bullet chars off each string element and
        adds in the appropriate bullet specified by the style.

        :param xs: list of strings
        :param style: bullet style to strip (dash, star, number)
        :return: a string with embedded newlines
    """
    stripped_xs = []
    for i, x in enumerate(xs):
        if NUM_BULLET_PATTERM.match(x):
            stripped_x = NUM_BULLET_PATTERM.match(x).group(2)
        elif GEN_BULLET_PATTERN.match(x):
            stripped_x = GEN_BULLET_PATTERN.match(x).group(1)
        else:
            stripped_x = x
        if style == "number":
            stripped_x = f"{i+1}. {stripped_x}"
        elif style in ["dash", "star"]:
            bullet = STYLE_TO_CHAR.get(style, "-")
            stripped_x = f"{bullet} {stripped_x}"
        else:
            pass
        stripped_xs.append(stripped_x)
    return "\n".join(stripped_xs)


def string_to_bool(s: str, choices: List[str]) -> bool:
    """ Convert a string to a boolean given list of choices representing
        True and False response

        :param s: string to convert
        :param choices: list of strings ["True", "False]
        :return: boolean
    """
    matches = [i for i, c in enumerate(choices)
                if re.search(c.lower(), s.lower()) is not None]
    if len(matches) == 0:
        return False
    else:
        return True if matches[0] == 0 else False

# xs = [
#     "This is a sentence",
#     "This is another sentence",
#     "This is a third sentence",
#     "This is a fourth sentence",
#     "This is our last sentence"
# ]
# print("-- list to str --")
# text = list_to_string(xs, style="number")
# print(text)

# print("-- str to list --")
# strlist = string_to_list(text, style="number")
# for s in strlist:
#     print(s)

# print("--- full circle ---")
# text2 = list_to_string(strlist, style="dash")
# print(text2)
