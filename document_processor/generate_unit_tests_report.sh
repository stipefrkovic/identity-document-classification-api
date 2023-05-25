pytest --cov --cov-config=.coveragerc --cov-report xml:reports/coverage.xml --junitxml=reports/pytest.xml

# The paths to replace
old_path="src/"
new_path="document_processor\/src\/"

# The coverage report file
file="reports/coverage.xml"

# Use sed to perform the replacement
sed -i "s#${old_path}#${new_path}#g" "${file}"
