#!/usr/bin/env bash

if [ $6 = "true" ]; then
rm -rf "$HOME/files${3}"
cd ../MPCAnonymousBloggingClient && ./MPCAnonymousBloggingClient -server ${1} -numServers ${2} -numClients ${3} \
   -fieldType ZpMersenne31 -l ${4}
fi

cd ../ && ./MPCAnonymousBloging -partyID ${1} -numServers ${2} -numClients ${3} \
   -fieldType ZpMersenne31 -l ${4} -partiesFile parties.conf  -internalIterationsNumber 1 -numThreads ${5}
