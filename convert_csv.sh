#!/bin/bash
cd `dirname $0`
cd datasets
IFS=$(echo -en "\n\b")
for file in $(ls *.csv); do
   LC_ALL=en_US ssconvert "$file" "${file%.csv}.xlsx"
done

