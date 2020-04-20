#! /bin/bash
for i in $(seq "${1}" 1 "${2}");
do
	./MPCAnonymousBloging  -l 44 -partyID "${i}" -numServers "${3}" -partiesNumber "${3}" -numClients "${4}"  -fieldType ZpMersenne31 \
	 -partiesFile Parties.txt  -internalIterationsNumber 1 -numThreads "${5}" -T "${6}" -toUmount "${7}" &
	echo "Running $i..."
done