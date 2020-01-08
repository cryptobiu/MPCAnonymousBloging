#!/usr/bin/env bash

if [ "${9}" = "true" ]; then
rm -rf ~/files*
cd ../MPCAnonymousBloggingClient && ./MPCAnonymousBloggingClient -server "${1}" -numServers "${2}" -numClients "${3}" \
   -fieldType ZpMersenne31 -l "${4}" -T "${8}" -toMount "${10}"
fi

./MPCAnonymousBloging -partyID "${1}" -numServers "${2}" -numClients "${3}" \
   -fieldType ZpMersenne31 -l "${4}" -partiesFile "${5}" -internalIterationsNumber "${6}" -numThreads "${7}" \
    -T "${8}" -toUmount "${10}"
