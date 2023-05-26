# Run the tests and generate reports
pytest --cov --cov-config=.coveragerc --cov-report xml:reports/coverage.xml --junitxml=reports/pytest.xml

# Fix paths in coverage.xml
old_path="src/"
new_path="document_processor\/src\/"

# The coverage.xml filepath
file="reports/coverage.xml"

# Use sed to perform the replacement
sed -i "s#${old_path}#${new_path}#g" "${file}"


# Fix paths in pytest.xml
old_path="src\."
new_path="document_processor\.src\."

# The pytest.xml filepath
file="reports/pytest.xml"

# Use sed to perform the replacement
sed -i "s#${old_path}#${new_path}#g" "${file}"


