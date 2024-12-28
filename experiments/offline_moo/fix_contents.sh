FILENAME=$1
SEARCH=$2
REPLACE=$3

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 filename 'search_string' 'replace_string'"
    exit 1
fi

sed -i "s/$SEARCH/$REPLACE/g" $FILENAME