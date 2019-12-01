#! /bin/bash
for i in $(seq "${1}" 1 "${2}");
do
	./MATRIX/run.sh "${i}" "${3}" "${4}" "${5}" "${6}" "${7}" "${8}" "${9}" "${10}" &
	echo "Running $i..."
done
