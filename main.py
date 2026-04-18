# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# zero-shot prompt
'''
You are given the following Python function. 
Generate a comprehensive set of unit tests for this function using pytest. 
The tests should cover normal cases, edge cases, and potential error conditions.

Function:
def copy(self):
    p = PreparedRequest()
    p.method = self.method
    p.url = self.url
    p.headers = self.headers.copy() if self.headers is not None else None
    p._cookies = _copy_cookie_jar(self._cookies)
    p.body = self.body
    p.hooks = self.hooks
    p._body_position = self._body_position
    return p

Generate only the test code. Do not include explanations.

'''

#few-shot prompt
'''
You are given the Python function. 
Generate a comprehensive set of unit tests for this function using pytest.
The tests should cover normal cases, edge cases, and potential error conditions.

Here are two examples:

Example 1:
Function:
def requote_uri(uri):
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        return quote(uri, safe=safe_without_percent)

Tests:
def test_requote_uri_already_quoted():
    assert requote_uri("http://example.com/path%20here") == "http://example.com/path%20here"

def test_requote_uri_unquoted_chars():
    result = requote_uri("http://example.com/path with spaces")
    assert " " not in result

def test_requote_uri_invalid_percent_sequence():
    # 含非法百分号序列时回退到直接 quote
    result = requote_uri("http://example.com/path%ZZhere")
    assert result is not None
    assert "%" in result

Example 2:
Function:
def _format_token(self, dt: datetime, token: Optional[str]) -> Optional[str]:
    if token and token.startswith("[") and token.endswith("]"):
        return token[1:-1]
    if token == "YYYY":
        return self.locale.year_full(dt.year)
    if token == "YY":
        return self.locale.year_abbreviation(dt.year)
    if token == "MM":
        return f"{dt.month:02d}"

Tests:
from arrow.formatter import DateTimeFormatter

@pytest.fixture
def formatter():
    return DateTimeFormatter("en_us")

def test_format_token_literal_bracket(formatter):
    dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
    assert formatter._format_token(dt, "[hello]") == "hello"

def test_format_token_year_full(formatter):
    dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
    assert formatter._format_token(dt, "YYYY") == "2024"

def test_format_token_month_zero_padded(formatter):
    dt = datetime(2024, 3, 1, tzinfo=timezone.utc)
    assert formatter._format_token(dt, "MM") == "03"

def test_format_token_none_input(formatter):
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert formatter._format_token(dt, None) is None

def test_format_token_unknown_token(formatter):
    dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert formatter._format_token(dt, "INVALID") is None

Now generate tests for the following function:

Function:
def copy(self):
    p = PreparedRequest()
    p.method = self.method
    p.url = self.url
    p.headers = self.headers.copy() if self.headers is not None else None
    p._cookies = _copy_cookie_jar(self._cookies)
    p.body = self.body
    p.hooks = self.hooks
    p._body_position = self._body_position
    return p

Generate only the test code. Do not include explanations.
'''

# Role-based prompt
'''
You are an experienced software test engineer with expertise in Python
and pytest. Your responsibility is to write thorough, well-structured
unit tests that accurately verify the behaviour of Python functions,
including handling of edge cases and error conditions.

You are given the Python function. Generate a comprehensive set of unit 
tests for this function using pytest.
The tests should cover normal cases, edge cases, and potential error conditions.


Function:
def copy(self):
    p = PreparedRequest()
    p.method = self.method
    p.url = self.url
    p.headers = self.headers.copy() if self.headers is not None else None
    p._cookies = _copy_cookie_jar(self._cookies)
    p.body = self.body
    p.hooks = self.hooks
    p._body_position = self._body_position
    return p

Generate only the test code. Do not include explanations.
'''

# Chain-of-thought Prompt

'''
You are given the following Python function. Before writing any test code,
reason step by step about the function's behaviour:

Step 1: Describe what the function is supposed to do.
Step 2: Identify the key input parameters and their expected types and ranges.
Step 3: List the normal cases that should be tested.
Step 4: Identify edge cases and boundary conditions.
Step 5: Consider what error or exception conditions might arise.

After completing your reasoning, generate a comprehensive set of unit 
tests for this function using pytest.
The tests should cover normal cases, edge cases, and potential error conditions.

Function:
def copy(self):
    p = PreparedRequest()
    p.method = self.method
    p.url = self.url
    p.headers = self.headers.copy() if self.headers is not None else None
    p._cookies = _copy_cookie_jar(self._cookies)
    p.body = self.body
    p.hooks = self.hooks
    p._body_position = self._body_position
    return p
'''