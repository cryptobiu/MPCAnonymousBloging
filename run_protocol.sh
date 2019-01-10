#! /bin/bash
for i in `seq $1 1 $2`;
do	
	./MPCAnonymousBloging  -l 3 -partyID $i -numServers $3 -numClients $4   -fieldType ZpMersenne31 -partiesFile Parties.txt  -internalIterationsNumber 1 &
	echo "Running $i..."
done
