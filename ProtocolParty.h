#ifndef PROTOCOLPARTY_H_
#define PROTOCOLPARTY_H_

#include <stdlib.h>

#include <libscapi/include/primitives/Matrix.hpp>
#include <libscapi/include/cryptoInfra/Protocol.hpp>
#include <libscapi/include/circuits/ArithmeticCircuit.hpp>
#include <libscapi/include/infra/Measurement.hpp>
#include <vector>
#include <bitset>
#include <iostream>
#include <fstream>
#include <chrono>
#include <libscapi/include/primitives/Mersenne.hpp>
#include <libscapi/include/comm/MPCCommunication.hpp>
#include <libscapi/include/infra/Common.hpp>
#include <libscapi/include/primitives/Prg.hpp>
#include <emmintrin.h>
#include <thread>
#include <libscapi/include/primitives/HashOpenSSL.hpp>
#include <omp.h>
#ifdef __NVCC__
#include "cudaGemm.h"
#include "utils.h"
#include <cuda_runtime.h>
#endif
#include <algorithm>


#define flag_print false
#define flag_print_timings true


using namespace std;
using namespace std::chrono;

template <class FieldType>
class ProtocolParty : public MPCProtocol{

private:

    /**
     * N - number of parties
     * M - number of gates
     * T - number of malicious
     */

    int N, M, T, m_partyId;//number of servers
    int times; //number of times to run the run function
    int iteration; //number of the current iteration


    vector<long> shiftbyOne;



    //
    long l;
    long  numClients;
    int numServers;
    int securityParamter = 40;
    long sqrtR;
    long sqrtU;
    int numThreads;

    vector<PrgFromOpenSSLAES> prgs;

    long batchSize;

    vector<FieldType> msgsVectorsFlat;
    vector<FieldType> squaresVectorsFlat;
    vector<FieldType> countersVectorsFlat;
    vector<FieldType> unitVectorsFlat;

    vector<FieldType> sumOfElementsVecs;
    vector<FieldType> openedSumOfElementsVecs;
    vector<FieldType> bigRVec;

    Measurement* timer;
    VDM<FieldType> matrix_vand;
    TemplateField<FieldType> *field;
//    vector<shared_ptr<ProtocolPartyData>>  parties;
    vector<FieldType> randomTAnd2TShares;
    vector<FieldType> randomSharesArray;
    vector<FieldType> bigR;
    vector<byte> h;//a string accumulated that should be hashed in the comparing views function.

    int offset = 0;
    int randomSharesOffset = 0;

    string s;
    vector<FieldType> beta;
    HIM<FieldType> matrix_for_interpolate;
    HIM<FieldType> matrix_for_t;
    HIM<FieldType> matrix_for_2t;
    vector<FieldType> y_for_interpolate;


    HIM<FieldType> matrix_him;

    VDMTranspose<FieldType> matrix_vand_transpose;

    HIM<FieldType> m;

    boost::asio::io_service io_service;
    vector<FieldType> alpha; // N distinct non-zero field elements


    thread t;
    bool toUmount;

public:

    ProtocolParty(int argc, char* argv[]);


    void roundFunctionSync(vector<vector<byte>> &sendBufs, vector<vector<byte>> &recBufs, int round);
    void exchangeData(vector<vector<byte>> &sendBufs,vector<vector<byte>> &recBufs, int first, int last);

    void roundFunctionSyncElements(vector<vector<FieldType>> &sendBufs, vector<vector<FieldType>> &recBufs, int round);
    void exchangeDataElements(vector<vector<FieldType>> &sendBufs,vector<vector<FieldType>> &recBufs, int first, int last);

    /**
     * This method runs the protocol:
     * 1. Preparation Phase
     * 2. Input Phase
     * 3. Computation Phase
     * 4. Verification Phase
     * 5. Output Phase
     */
    void run() override;

    bool hasOffline() {
        return true;
    }


    bool hasOnline() override {
        return true;
    }

    /**
     * This method runs the protocol:
     * Preparation Phase
     */
    void runOffline() override;

    /**
     * This method runs the protocol:
     * Input Phase
     * Computation Phase
     * Verification Phase
     * Output Phase
     */
    void runOnline() override;


    /**
     * We describe the protocol initialization.
     * In particular, some global variables are declared and initialized.
     */
    void initializationPhase();


    /**
     * A random double-sharing is a pair of two sharings of the same random value, where the one sharing is
     * of degree t, and the other sharing is of degree 2t. Such random double-sharing are of big help in the
     * multiplication protocol.
     * We use hyper-invertible matrices to generate random double-sharings. The basic idea is as follows:
     * Every party generates one random double-sharing. These n double-sharings are processes through a
     * hyper-invertible matrix. From the resulting n double-sharings, t are checked to be valid (correct degree,
     * same secret), and t are then kept as “good” double-sharings. This is secure due to the diversion property
     * of hyper-invertible matrix: We know that n − t of the input double-sharings are good. So, if there are t
     * valid output double-sharings, then all double-sharings must be valid. Furthermore, the adversary knows
     * his own up to t input double-sharings, and learns t output double sharings. So, n − 2t output double
     * sharings are random and unknown to the adversary.
     * For the sake of efficiency, we do not publicly reconstruct t of the output double-sharings. Rather, we
     * reconstruct 2t output double sharings, each to one dedicated party only. At least t of these parties are
     * honest and correctly validate the reconstructed double-sharing.
     *
     * The goal of this phase is to generate “enough” double-sharings to evaluate the circuit. The double-
     * sharings are stored in a buffer SharingBuf , where alternating a degree-t and a degree-2t sharing (of the same secret)
     * is stored (more precisely, a share of each such corresponding sharings is stored).
     * The creation of double-sharings is:
     *
     * Protocol Generate-Double-Sharings:
     * 1. ∀i: Pi selects random value x-(i) and computes degree-t shares x1-(i) and degree-2t shares x2-(i).
     * 2. ∀i,j: Pi sends the shares x1,j and X2,j to party Pj.
     * 3. ∀j: Pj applies a hyper-invertible matrix M on the received shares, i.e:
     *      (y1,j,..., y1,j) = M(x1,j,...,x1,j)
     *      (y2,j,...,y2,j) = M (x2,j,...,x2,)
     * 4. ∀j, ∀k ≤ 2t: Pj sends y1,j and y2,j to Pk.
     * 5. ∀k ≤ 2t: Pk checks:
     *      • that the received shares (y1,1,...,y1,n) are t-consistent,
     *      • that the received shares (y2,1,...,y2,n) are 2t-consistent, and
     *      • that both sharings interpolate to the same secret.
     *
     * We use this algorithm, but extend it to capture an arbitrary number of double-sharings.
     * This is, as usual, achieved by processing multiple buckets in parallel.
     */
    bool preparationPhase();

    void inputPhase();
    /**
     * This protocol is secure only in the presence of a semi-honest adversary.
     */
    void DNHonestMultiplication(FieldType *a, FieldType *b, vector<FieldType> &cToFill, int numOfTrupples);

    void readclientsinputs(vector<FieldType> &msgsVectorsFlat, vector<FieldType> &squaresVectorsFlat, vector<FieldType> &countersVectorsFlat, vector<FieldType> &unitVectorsFlat);
    void readServerFile(string fileName, FieldType* msg, FieldType* squares, FieldType* counters, FieldType* unitVector, FieldType * e);

    int validMsgsTestFlat(vector<FieldType> &msgsVectors, vector<FieldType> &msgsVectorsSquares, vector<FieldType> & counters, vector<FieldType> &unitVectors);

    int fasterUnitVectorsTestFlat(vector<FieldType> &vecs, int size,FieldType *randomElements);


    void generateSharedMatricesOptimizedFlat(vector<FieldType> &shiftedMsgsVectors,
                                        vector<FieldType> &shiftedMsgsVectorsSquares,
                                        vector<FieldType> &shiftedMsgsVectorsCounters,
                                        vector<FieldType> &shiftedUnitVectors,
                                        vector<FieldType> &accMsgsMat,
                                        vector<FieldType> &accMsgsSquareMat,
                                        vector<FieldType> &accCountersMat);


    void matrixMulTN(FieldType *C, int ldc, const FieldType *A, int lda, const FieldType *B, int ldb,
            int hA, int wA, int wB);

    void regMatrixMulTN(FieldType *C, FieldType *A, int rowa, int cola, FieldType *B, int rowb,  int colb);



//    void multiplyVectors(vector<vector<FieldType>> & input,
//                            vector<vector<FieldType>> & unitVectors,
//                            vector<FieldType> & output,
//                            int numOfRows,
//                            int numOfCols);
//

    void multMatricesFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors,
                        vector<long> & outputDouble, int newNumRows, int newNumCols, int i, __m256i mask, bool toReduce, __m256i & p);

    void reduce(__m256i & output0, __m256i & output1, __m256i & output2, __m256i & output3, __m256i & output4,
            __m256i & output5, __m256i & output6, __m256i & output7, __m256i & p);
    void reduceMatrix(vector<long> & outputDouble, int newNumRows, int newNumCols, __m256i mask, __m256i p);

//    void multiplyVectorsWithThreads(vector<vector<FieldType>> & input, vector<vector<FieldType>> & unitVectors,
//                                    vector<FieldType> & output, int numOfRows, int numOfCols);
//
//    void multiplyVectorsPerThread(vector<vector<FieldType>> & input, vector<vector<FieldType>> & unitVectors, vector<long> & outputDouble,
//                                                            int newNumRows, int newNumCols, int start, int end);

    void multiplyVectorsWithThreadsFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors,
                                    vector<FieldType> & output, int numOfRows, int numOfCols);

    void multiplyVectorsPerThreadFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors, vector<long> & outputDouble,
                                  int newNumRows, int newNumCols, int start, int end);

    void prepareVecsForUnitTestThreads(vector<vector<FieldType>> & randomVecs, vector<FieldType> & vecs, vector<FieldType> & messagesShares,
                                       vector<FieldType>& msgByRandomSum, vector<FieldType>& VecByRandomSum, FieldType* randomElements, int size, int start, int end);

        void assignSumsPerThreadFlat(vector<long> & sum01, vector<FieldType> & vecs, int size, byte* constRandomBitsPrim,
                                                        vector<vector<FieldType>> & randomVecs, int start, int end);
    int generateClearMatricesForTesting(vector<FieldType> &accMsgsMat,
                                        vector<FieldType> &accMsgsSquareMat,
                                        vector<FieldType> &accCountersMat,
                                        vector<int> &accIntCountersMat);

    int generateClearMatrices(vector<FieldType> &accMats, vector<FieldType> &accFieldCountersMat,vector<int> &accIntCountersMat);

    void generateRandomShiftingindices(vector<int> &randomShiftingVec);

    void splitShiftFlat(vector<FieldType> &msgsVectors, vector<FieldType> &squaresVectors, vector<FieldType> &countersVectors, vector<FieldType> &unitVectors);

    void splitShiftByThreads(vector<int> & randomShiftingIndices, vector<FieldType> & originalArr, long size, long l, long position, long start, long end);

//        void copyBackToVectors();



    void commitOnMatrices(vector<FieldType> &accMats, vector<FieldType> &accFieldCountersMat,
                                                   vector<vector<byte>> &recBufsBytes);

    void extractMessagesForTesting(vector<FieldType> &accMsgsMat,
                         vector<FieldType> &accMsgsSquareMat,
                         vector<int> &accIntCountersMat, int numMsgs);

    void calcPairMessages(FieldType & a, FieldType & b, int counter, int i, int j);

    void printOutputMessagesForTesting(vector<FieldType> &accMsgsMat,
                                                                 vector<FieldType> &accMsgsMat2,
                                                                 vector<int> &accIntCountersMat, int numMsgs);

    void offlineDNForMultiplication(int numOfTriples);


    /**
     * The input phase proceeds in two steps:
     * First, for each input gate, the party owning the input creates shares for that input by choosing a random coefficients for the polynomial
     * Then, all the shares are sent to the relevant party
     */


    void prepareForUnitTestFlat(vector<FieldType> &randomElements, vector<FieldType> &msgsVectors, vector<FieldType> &msgsVectorsSquares, vector<FieldType> & counters,
                                                      vector<FieldType> &sumXandSqaure, vector<FieldType> &msgsVectorsForUnitTest, int start, int end);
    void generateRandomShares(int numOfRandoms, vector<FieldType> &randomElementsToFill);
    void getRandomShares(int numOfRandoms, vector<FieldType> &randomElementsToFill);
    void generateRandomSharesWithCheck(int numOfRnadoms, vector<FieldType>& randomElementsToFill);
    void generateRandom2TAndTShares(int numOfRandomPairs, vector<FieldType>& randomElementsToFill);

    void calcSendBufElements(vector<vector<FieldType>> & sendBufsElements, PrgFromOpenSSLAES & prg, int start, int end);
    void calcRecBufElements(vector<vector<FieldType>> & recBufsElements, vector<FieldType> & randomElementsToFill, int start, int end);
    /**
     * Check whether given points lie on polynomial of degree d.
     * This check is performed by interpolating x on the first d + 1 positions of α and check the remaining positions.
     */
    bool checkConsistency(vector<FieldType>& x, int d);

    FieldType reconstructShare(vector<FieldType>& x, int d);

    void openShare(int numOfRandomShares, vector<FieldType> &Shares, vector<FieldType> &secrets, int d);

    void openShareSetRecBuf(int numOfRandomShares, vector<FieldType> &Shares, vector<FieldType> &secrets,
                                                      int d, vector<vector<byte>> &recBufsBytes);


    /**
     * The cheap way: Create a HIM from the αi’s onto ZERO (this is actually a row vector), and multiply
     * this HIM with the given x-vector (this is actually a scalar product).
     * The first (and only) element of the output vector is the secret.
     */
    FieldType interpolate(vector<FieldType>& x);


    /**
     * Walk through the circuit and verify the multiplication gates.
     * We first generate the random elements using a common AES key that was generated by the parties,
     * run the relevant verification algorithm and return accept/reject according to the output
     * of the verification algorithm.
     */
    int verificationPhase();

    vector<byte> generateCommonKey();
    void generatePseudoRandomElements(vector<byte> & aesKey, vector<FieldType> &randomElementsToFill, int numOfRandomElements);

    /**
     * Walk through the circuit and reconstruct output gates.
     */
    void outputPhase();

    ~ProtocolParty();


    void batchConsistencyCheckOfShares(const vector<FieldType> &inputShares);


};




template <class FieldType>
ProtocolParty<FieldType>::ProtocolParty(int argc, char* argv[]) : MPCProtocol("MPCAnonymuosBlogging", argc, argv)
{

    l = stoi(this->getParser().getValueByKey(arguments, "l"));
    m_partyId = stoi(this->getParser().getValueByKey(arguments, "partyID"));

    numServers = stoi(this->getParser().getValueByKey(arguments, "numServers"));
    numClients = stoi(this->getParser().getValueByKey(arguments, "numClients"));
    numThreads = stoi(this->getParser().getValueByKey(arguments, "numThreads"));

    string fieldType = this->getParser().getValueByKey(arguments, "fieldType");

    this->times = stoi(this->getParser().getValueByKey(arguments, "internalIterationsNumber"));
    toUmount = (this->getParser().getValueByKey(arguments, "toUmount").compare("true") == 0);

    vector<string> subTaskNames{"Offline", "preparationPhase", "Online", "inputPhase", "ComputePhase",
                                "VerificationPhase", "outputPhase", "generateSharedMatricesOptimized",
                                "extractMessagesForTesting", "printOutputMessagesForTesting"};
    timer = new Measurement(*this, subTaskNames);

    if(fieldType.compare("ZpMersenne31") == 0) {
        field = new TemplateField<FieldType>(2147483647);
    } else if(fieldType.compare("ZpMersenne61") == 0) {
        field = new TemplateField<FieldType>(0);
    }


    N = numServers;
    T = stoi(this->getParser().getValueByKey(arguments, "T"));

    sqrtR = (int)((sqrt(l*2.7 * numClients)))/l+1;
    sqrtU = (int)(sqrt(l*2.7 * numClients))+1;


    batchSize = numClients;
    s = to_string(m_partyId);

//    MPCCommunication comm;
//    string partiesFile = this->getParser().getValueByKey(arguments, "partiesFile");
//
//    parties = comm.setCommunication(io_service, m_partyId, N, partiesFile);

    prgs.resize(numThreads);
    int* keyBytes = new int[4];
    for (int i=0; i<numThreads; i++){
        for (int j=0; j<4; j++){
            keyBytes[j] = field->Random().elem;
        }
        SecretKey key((byte*)keyBytes, 16, "");
        prgs[i].setKey(key);
    }
    delete [] keyBytes;

//
//
//
//
    auto t1 = high_resolution_clock::now();
    initializationPhase(/*matrix_him, matrix_vand, m*/);

    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds initializationPhase: " << duration << endl;
    }

    shiftbyOne.resize(securityParamter);
    for(int i=0; i<securityParamter; i++){
        shiftbyOne[i] = 1 << i;
    }
}


template <class FieldType>
void ProtocolParty<FieldType>::run() {

    for (iteration=0; iteration<times; iteration++){

        auto t1start = high_resolution_clock::now();
        timer->startSubTask("Offline", iteration);
        runOffline();
        timer->endSubTask("Offline", iteration);
        timer->startSubTask("Online", iteration);
        runOnline();
        timer->endSubTask("Online", iteration);

        auto t2end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(t2end-t1start).count();

        cout << "time in milliseconds for protocol: " << duration << endl;
    }


}

template <class FieldType>
void ProtocolParty<FieldType>::runOffline() {
    auto t1 = high_resolution_clock::now();
    timer->startSubTask("preparationPhase", iteration);
    if(preparationPhase() == false) {
        if(flag_print) {
            cout << "cheating!!!" << '\n';}
        return;
    }
    else {
        if(flag_print) {
            cout << "no cheating!!!" << '\n' << "finish Preparation Phase" << '\n';}
    }
    timer->endSubTask("preparationPhase", iteration);
    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds preparationPhase: " << duration << endl;
    }
}

template <class FieldType>
void ProtocolParty<FieldType>::runOnline() {

    auto t1 = high_resolution_clock::now();
    timer->startSubTask("inputPhase", iteration);
    inputPhase();
    timer->endSubTask("inputPhase", iteration);
    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2-t1).count();

    if(flag_print_timings) {
        cout << "time in milliseconds inputPhase: " << duration << endl;
    }

    t1 = high_resolution_clock::now();
    timer->startSubTask("VerificationPhase", iteration);
   // t = thread(&ProtocolParty::verificationPhase, this);
    auto flag = verificationPhase();
    timer->endSubTask("VerificationPhase", iteration);
    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t2-t1).count();

    if(flag_print_timings) {
        cout << "time in milliseconds verificationPhase: " << duration << endl;
    }

    t1 = high_resolution_clock::now();
    timer->startSubTask("outputPhase", iteration);
    outputPhase();
    timer->endSubTask("outputPhase", iteration);
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();

    if(flag_print_timings) {
        cout << "time in milliseconds outputPhase: " << duration << endl;
    }

    //t.join();

}

template <class FieldType>
void ProtocolParty<FieldType>::batchConsistencyCheckOfShares(const vector<FieldType> &inputShares) {//first generate the common aes key


    auto key = generateCommonKey();

    //print key
    if (flag_print) {
        for (int i = 0; i < key.size(); i++) {
            cout << "key[" << i << "] for party :" << m_partyId << "is : " << (int) key[i] << endl;
        }
    }

    //calc the number of times we need to run the verification -- ceiling
    int iterations = (5 + field->getElementSizeInBytes() - 1) / field->getElementSizeInBytes();

    vector<FieldType> randomElements(inputShares.size()*iterations);
    generatePseudoRandomElements(key, randomElements, inputShares.size());


    for(int j=0; j<iterations;j++) {
        vector<FieldType> r(1);//vector holding the random shares generated
        vector<FieldType> v(1);
        vector<FieldType> secret(1);


        getRandomShares(1, r);

        for (int i = 0; i < inputShares.size(); i++)
            v[0] += randomElements[i+j*inputShares.size()] * inputShares[i];

        v[0] += r[0];


        //if all the the parties share lie on the same polynomial this will not abort
        openShare(1, v, secret, T);
    }
}


template <class FieldType>
void ProtocolParty<FieldType>::generateRandomSharesWithCheck(int numOfRandoms, vector<FieldType>& randomElementsToFill){


    getRandomShares(numOfRandoms, randomElementsToFill);

    batchConsistencyCheckOfShares(randomElementsToFill);

}

template <class FieldType>
void ProtocolParty<FieldType>::generateRandomShares(int numOfRandoms, vector<FieldType> &randomElementsToFill) {
    int index = 0;
    int robin = 0;
    int no_random = numOfRandoms;

    vector<FieldType> x1(N),y1(N), x2(N),y2(N), t1(N), r1(N), t2(N), r2(N);

    vector<vector<FieldType>> sendBufsElements(N);
    vector<vector<FieldType>> recBufsElements(N);

    // the number of buckets (each bucket requires one double-sharing
    // from each party and gives N-2T random double-sharings)
    int no_buckets = (no_random / (N - T)) + 1;

    //sharingBufTElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings
    //sharingBuf2TElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings

    //maybe add some elements if a partial bucket is needed
    randomElementsToFill.resize(no_buckets*(N - T));


    for(int i=0; i < N; i++)
    {
        sendBufsElements[i].resize(no_buckets);
        recBufsElements[i].resize(no_buckets);
    }

    /**
     *  generate random sharings.
     *  first degree t.
     *
     */
    for(int k=0; k < no_buckets; k++)
    {
        // generate random degree-T polynomial
        for(int i = 0; i < T + 1; i++)
        {
            // A random field element, uniform distribution, note that x1[0] is the secret which is also random
            x1[i] = field->Random();

        }

        matrix_vand.MatrixMult(x1, y1, T + 1); // eval poly at alpha-positions

        // prepare shares to be sent
        for(int i=0; i < N; i++)
        {
            //cout << "y1[ " <<i<< "]" <<y1[i] << endl;
            sendBufsElements[i][k] = y1[i];

        }
    }

    if(flag_print) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < sendBufsElements[0].size(); k++) {

                // cout << "before roundfunction4 send to " <<i <<" element: "<< k << " " << sendBufsElements[i][k] << endl;
            }
        }
        cout << "sendBufs" << endl;
        cout << "N" << N << endl;
        cout << "T" << T << endl;
    }

    roundFunctionSyncElements(sendBufsElements, recBufsElements, 4);


    for(int k=0; k < no_buckets; k++) {
        for (int i = 0; i < N; i++) {
            t1[i] = recBufsElements[i][k];

        }
        matrix_vand_transpose.MatrixMult(t1, r1, N - T);

        //copy the resulting vector to the array of randoms
        for (int i = 0; i < N - T; i++) {

            randomElementsToFill[index] = r1[i];
            index++;

        }
    }
}
template <class FieldType>
void ProtocolParty<FieldType>::getRandomShares(int numOfRandoms, vector<FieldType> &randomElementsToFill){

    randomElementsToFill.assign (randomSharesArray.begin() + randomSharesOffset,
                                 randomSharesArray.begin() + randomSharesOffset + numOfRandoms);

    randomSharesOffset += numOfRandoms;

}

template <class FieldType>
void ProtocolParty<FieldType>::generateRandom2TAndTShares(int numOfRandomPairs, vector<FieldType>& randomElementsToFill){
    auto t1 = high_resolution_clock::now();
    int robin = 0;
    int no_random = numOfRandomPairs;

    // the number of buckets (each bucket requires one double-sharing
    // from each party and gives N-2T random double-sharings)
    int no_buckets = (no_random / (N-T))+1;

    vector<vector<FieldType>> sendBufsElements(N, vector<FieldType>(no_buckets*2));
    vector<vector<FieldType>> recBufsElements(N, vector<FieldType>(no_buckets*2));

    //maybe add some elements if a partial bucket is needed
    randomElementsToFill.resize(no_buckets*(N-T)*2);
    vector<FieldType> randomElementsOnlyTshares (no_buckets*(N-T) );

    int sizeForEachThread;
    if (no_buckets <= numThreads){
        numThreads = no_buckets;
        sizeForEachThread = 1;
    } else{
        sizeForEachThread = (no_buckets + numThreads - 1)/ numThreads;
    }
    vector<thread> threads(numThreads);

    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds header: " << duration << endl;
    }
     t1 = high_resolution_clock::now();




    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * sizeForEachThread <= no_buckets) {
            threads[t] = thread(&ProtocolParty::calcSendBufElements, this, ref(sendBufsElements), ref(prgs[t]), t * sizeForEachThread, (t + 1) * sizeForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::calcSendBufElements, this, ref(sendBufsElements), ref(prgs[t]), t * sizeForEachThread, no_buckets);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

     t2 = high_resolution_clock::now();

     duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds calcSendBufElements: " << duration << endl;
    }


    t1 = high_resolution_clock::now();


    roundFunctionSyncElements(sendBufsElements, recBufsElements,4);
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds round function: " << duration << endl;
    }

    t1 = high_resolution_clock::now();

    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * sizeForEachThread <= no_buckets) {
            threads[t] = thread(&ProtocolParty::calcRecBufElements, this, ref(recBufsElements), ref(randomElementsToFill), t * sizeForEachThread, (t + 1) * sizeForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::calcRecBufElements, this, ref(recBufsElements), ref(randomElementsToFill), t * sizeForEachThread, no_buckets);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds calcRecBufElements: " << duration << endl;
    }

    //check validity of the t-shares. 2t-shares do not have to be checked
    //copy the t-shares for checking
    t1 = high_resolution_clock::now();
    for(int i=0; i<randomElementsOnlyTshares.size(); i++){

        randomElementsOnlyTshares[i] = randomElementsToFill[2*i];
    }

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds copy: " << duration << endl;
    }

    t1 = high_resolution_clock::now();
    batchConsistencyCheckOfShares(randomElementsOnlyTshares);

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds batch consistency: " << duration << endl;
    }

}

template <class FieldType>
void ProtocolParty<FieldType>::calcSendBufElements(vector<vector<FieldType>> & sendBufsElements, PrgFromOpenSSLAES & prg, int start, int end){

    vector<FieldType> x1(N),y1(N), x2(N),y2(N);

    int* tempInt;

    cout << "****************" << sendBufsElements.size() << endl;

    for(int k=start; k < end; k++)
    {
        // generate random degree-T polynomial
        tempInt = (int*)prg.getPRGBytesEX((T+1)*4);

        for(int i = 0; i < T+1; i++)
        {
            // A random field element, uniform distribution, note that x1[0] is the secret which is also random
            x1[i] = field->GetElement(tempInt[i]);

        }

        matrix_vand.MatrixMult(x1, y1,T+1); // eval poly at alpha-positions

        x2[0] = x1[0];

        // generate random degree-T polynomial
        tempInt = (int*)prg.getPRGBytesEX((2*T+1)*4);

        for(int i = 1; i < 2*T+1; i++)
        {
            // A random field element, uniform distribution, note that x1[0] is the secret which is also random
            x2[i] = field->GetElement(tempInt[i]);

        }

        matrix_vand.MatrixMult(x2, y2,2*T+1);

        // prepare shares to be sent
        for(int i=0; i < N; i++)
        {
            //cout << "y1[ " <<i<< "]" <<y1[i] << endl;
            sendBufsElements[i][2*k] = y1[i];
            sendBufsElements[i][2*k + 1] = y2[i];

        }
    }
}

template <class FieldType>
void ProtocolParty<FieldType>::calcRecBufElements(vector<vector<FieldType>> & recBufsElements, vector<FieldType> & randomElementsToFill, int start, int end){

    vector<FieldType> t1(N), r1(N), t2(N), r2(N);

    for(int k=start; k < end; k++) {
        for (int i = 0; i < N; i++) {

            t1[i] = recBufsElements[i][2*k];
            t2[i] = recBufsElements[i][(2*k +1)];


        }
        matrix_vand_transpose.MatrixMult(t1, r1,N-T);
        matrix_vand_transpose.MatrixMult(t2, r2,N-T);

        //copy the resulting vector to the array of randoms
        for (int i = 0; i < (N - T); i++) {

            randomElementsToFill[k*2] = r1[i];
            randomElementsToFill[k*2 +1] = r2[i];

        }
    }
}

/**
 * some global variables are initialized
 * @param GateValueArr
 * @param GateShareArr
 * @param matrix_him
 * @param matrix_vand
 * @param alpha
 */
template <class FieldType>
void ProtocolParty<FieldType>::initializationPhase()
{
    bigR.resize(1);

cout<<"requested size is "<<numClients*sqrtR*l<<endl;
//    msgsVectorsFlat.resize(numClients*sqrtR*l);
//    squaresVectorsFlat.resize(numClients*sqrtR*l);
//    countersVectorsFlat.resize(numClients*sqrtR);
//    unitVectorsFlat.resize(numClients*sqrtU);

    int uvSize = batchSize*sqrtU;
    if (uvSize % 8 != 0) {
        uvSize = (uvSize / 8)*8 + 8;
    }

    int size = batchSize*sqrtR*l;
    if (size % 64 != 0) {
        size = (size / 64)*64 + 64;
    }

    int cSize = batchSize*sqrtR;
    if (cSize % 64 != 0) {
        cSize = (cSize / 64)*64 + 64;
    }

//    msgsVectorsShiftedFlat.resize(size);
//    squaresVectorsShiftedFlat.resize(size);
//    countersVectorsShiftedFlat.resize(cSize);
//    unitVectorsShiftedFlat.resize(uvSize);


    msgsVectorsFlat.resize(size);
    squaresVectorsFlat.resize(size);
    countersVectorsFlat.resize(cSize);
    unitVectorsFlat.resize(uvSize);

//    sum1.resize(batchSize*securityParamter);
//    sum0.resize(batchSize*securityParamter);//do in a 1 dimension array for multiplication
//    sum01.resize(2*batchSize*securityParamter);//do in a 1 dimension array for multiplication

    sumOfElementsVecs.resize(batchSize*2, *field->GetZero());
    openedSumOfElementsVecs.resize(batchSize*2, *field->GetZero());


    beta.resize(1);
    y_for_interpolate.resize(N);

    alpha.resize(N); // N distinct non-zero field elements
    vector<FieldType> alpha1(N-T);
    vector<FieldType> alpha2(T);

    beta[0] = field->GetElement(0); // zero of the field
    matrix_for_interpolate.allocate(1,N, field);


    matrix_him.allocate(N,N,field);
    matrix_vand.allocate(N,N,field);
    matrix_vand_transpose.allocate(N,N,field);
    m.allocate(T, N-T,field);

    // Compute Vandermonde matrix VDM[i,k] = alpha[i]^k
    matrix_vand.InitVDM();
    matrix_vand_transpose.InitVDMTranspose();

    // Prepare an N-by-N hyper-invertible matrix
    matrix_him.InitHIM();

    // N distinct non-zero field elements
    for(int i=0; i<N; i++)
    {
        alpha[i]=field->GetElement(i+1);
    }

    for(int i = 0; i < N-T; i++)
    {
        alpha1[i] = alpha[i];
    }
    for(int i = N-T; i < N; i++)
    {
        alpha2[i - (N-T)] = alpha[i];
    }

    m.InitHIMByVectors(alpha1, alpha2);

    matrix_for_interpolate.InitHIMByVectors(alpha, beta);

    vector<FieldType> alpha_until_t(T + 1);
    vector<FieldType> alpha_from_t(N - 1 - T);

    // Interpolate first d+1 positions of (alpha,x)
    matrix_for_t.allocate(N - 1 - T, T + 1, field); // slices, only positions from 0..d
    //matrix_for_t.InitHIMByVectors(alpha_until_t, alpha_from_t);
    matrix_for_t.InitHIMVectorAndsizes(alpha, T+1, N-T-1);

    vector<FieldType> alpha_until_2t(2*T + 1);
    vector<FieldType> alpha_from_2t(N - 1 - 2*T);

    // Interpolate first d+1 positions of (alpha,x)
    matrix_for_2t.allocate(N - 1 - 2*T, 2*T + 1, field); // slices, only positions from 0..d
    //matrix_for_2t.InitHIMByVectors(alpha_until_2t, alpha_from_2t);
    matrix_for_2t.InitHIMVectorAndsizes(alpha, 2*T + 1, N-(2*T +1));


    if(flag_print){
        cout<< "matrix_for_t : " <<endl;
        matrix_for_t.Print();

        cout<< "matrix_for_2t : " <<endl;
        matrix_for_2t.Print();

    }




    auto t1 = high_resolution_clock::now();
    readclientsinputs(msgsVectorsFlat, squaresVectorsFlat, countersVectorsFlat, unitVectorsFlat);

    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds read clients inputs: " << duration << endl;
    }


    cout <<"sqrtR = "<<sqrtR<<endl;
    cout <<"sqrtU = "<<sqrtU<<endl;
}

template <class FieldType>
bool ProtocolParty<FieldType>::preparationPhase()
{
    int iterations =   (5 + field->getElementSizeInBytes() - 1) / field->getElementSizeInBytes();
    int keysize = 16/field->getElementSizeInBytes() + 1;

    int numOfRandomShares = 10*keysize + 1;
    randomSharesArray.resize(numOfRandomShares);

    randomSharesOffset = 0;
    //generate enough random shares for the AES key
    generateRandomShares(numOfRandomShares, randomSharesArray);


    //run offline for all the future multiplications including the multiplication of the protocol

    offset = 0;
    offlineDNForMultiplication(numClients*(4*l + securityParamter*2));



    //first generate numOfTriples random shares
    generateRandomSharesWithCheck(1, bigR);

    //set this random share to an entire array so we can use the semi honest multiplication
    bigRVec.resize(numClients*securityParamter);
    fill(bigRVec.begin(), bigRVec.end(), bigR[0]);


    return true;
}


/**
 * Check whether given points lie on polynomial of degree d. This check is performed by interpolating x on
 * the first d + 1 positions of α and check the remaining positions.
 */
template <class FieldType>
bool ProtocolParty<FieldType>::checkConsistency(vector<FieldType>& x, int d)
{
    if(d == T)
    {
        vector<FieldType> y(N - 1 - d); // the result of multiplication
        vector<FieldType> x_until_t(T + 1);

        for (int i = 0; i < T + 1; i++) {
            x_until_t[i] = x[i];
        }


        matrix_for_t.MatrixMult(x_until_t, y);

        // compare that the result is equal to the according positions in x
        for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
        {
            if ((y[i]) != (x[d + 1 + i])) {
                return false;
            }
        }
        return true;
    } else if (d == 2*T)
    {
        vector<FieldType> y(N - 1 - d); // the result of multiplication

        vector<FieldType> x_until_2t(2*T + 1);

        for (int i = 0; i < 2*T + 1; i++) {
            x_until_2t[i] = x[i];
        }

        matrix_for_2t.MatrixMult(x_until_2t, y);

        // compare that the result is equal to the according positions in x
        for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
        {
            if ((y[i]) != (x[d + 1 + i])) {
                return false;
            }
        }
        return true;

    } else {
        vector<FieldType> alpha_until_d(d + 1);
        vector<FieldType> alpha_from_d(N - 1 - d);
        vector<FieldType> x_until_d(d + 1);
        vector<FieldType> y(N - 1 - d); // the result of multiplication

        for (int i = 0; i < d + 1; i++) {
            alpha_until_d[i] = alpha[i];
            x_until_d[i] = x[i];
        }
        for (int i = d + 1; i < N; i++) {
            alpha_from_d[i - (d + 1)] = alpha[i];
        }
        // Interpolate first d+1 positions of (alpha,x)
        HIM<FieldType> matrix(N - 1 - d, d + 1, field); // slices, only positions from 0..d
        matrix.InitHIMByVectors(alpha_until_d, alpha_from_d);
        matrix.MatrixMult(x_until_d, y);

        // compare that the result is equal to the according positions in x
        for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
        {
            if (y[i] != x[d + 1 + i]) {
                return false;
            }
        }
        return true;
    }
    return true;
}

// Interpolate polynomial at position Zero
template <class FieldType>
FieldType ProtocolParty<FieldType>::interpolate(vector<FieldType>& x)
{
    matrix_for_interpolate.MatrixMult(x, y_for_interpolate);
    return y_for_interpolate[0];
}



template <class FieldType>
FieldType ProtocolParty<FieldType>::reconstructShare(vector<FieldType>& x, int d){

    if (!checkConsistency(x, d))
        exit(-1);
//        cout << "cheating reconstruct!!!" << endl;
    else
        return interpolate(x);
}



template <class FieldType>
void ProtocolParty<FieldType>::DNHonestMultiplication(FieldType *a, FieldType *b, vector<FieldType> &cToFill, int numOfTrupples) {

    int fieldByteSize = field->getElementSizeInBytes();
    vector<FieldType> xyMinusRShares(numOfTrupples);//hold both in the same vector to send in one batch
    vector<byte> xyMinusRSharesBytes(numOfTrupples *fieldByteSize);//hold both in the same vector to send in one batch

    vector<FieldType> xyMinusR;//hold both in the same vector to send in one batch
    vector<byte> xyMinusRBytes;

    vector<vector<FieldType>> recBufsElements(N);
    vector<vector<FieldType>> sendBufsElements(N);


    //generate the shares for x+a and y+b. do it in the same array to send once
    for (int k = 0; k < numOfTrupples; k++)//go over only the logit gates
    {
        //compute the share of xy-r
        xyMinusRShares[k] = a[k]*b[k] - randomTAnd2TShares[offset + 2*k+1];

    }

    //set the acctual number of mult gate proccessed in this layer
    int acctualNumOfMultGates = numOfTrupples;
    int numOfElementsForParties = acctualNumOfMultGates/N;
    int indexForDecreasingSize = acctualNumOfMultGates - numOfElementsForParties *N;

    int counter=0;
    int currentNumOfElements;
    for(int i=0; i<N; i++){

        currentNumOfElements = numOfElementsForParties;
        if(i<indexForDecreasingSize)
            currentNumOfElements++;

        //fill the send buf according to the number of elements to send to each party
        sendBufsElements[i].resize(currentNumOfElements);

        for(int j=0; j<currentNumOfElements; j++) {

            sendBufsElements[i][j] = xyMinusRShares[counter];
            counter++;

        }

    }

    //resize the recbuf array.
    int myNumOfElementsToExpect = numOfElementsForParties;
    if (m_partyId < indexForDecreasingSize) {
        myNumOfElementsToExpect = numOfElementsForParties + 1;
    }

    for(int i=0;i<N;i++){

        //recBufsBytes[i].resize(myNumOfElementsToExpect*fieldByteSize);
        recBufsElements[i].resize(myNumOfElementsToExpect);

    }


    roundFunctionSyncElements(sendBufsElements, recBufsElements,20);

    xyMinusR.resize(myNumOfElementsToExpect);
    xyMinusRBytes.resize(myNumOfElementsToExpect*fieldByteSize);

    //reconstruct the shares that I am responsible of recieved from the other parties
    vector<FieldType> xyMinurAllShares(N);

    for (int k = 0;k < myNumOfElementsToExpect; k++)//go over only the logit gates
    {
        for (int i = 0; i < N; i++) {

            xyMinurAllShares[i] = recBufsElements[i][k];
        }

        // reconstruct the shares by P0
        xyMinusR[k] = interpolate(xyMinurAllShares);

    }


    //prepare the send buffers
    for(int i=0; i<N; i++){
        //sendBufsBytes[i] = xyMinusRBytes;
        sendBufsElements[i] = xyMinusR;
    }


    for(int i=0; i<N; i++){

        currentNumOfElements = numOfElementsForParties;
        if(i<indexForDecreasingSize)
            currentNumOfElements++;

        //recBufsBytes[i].resize(currentNumOfElements* fieldByteSize);
        recBufsElements[i].resize(currentNumOfElements);

    }
    roundFunctionSyncElements(sendBufsElements, recBufsElements,21);


    xyMinusR.resize(acctualNumOfMultGates);
    counter = 0;

    for(int i=0; i<N; i++){

        currentNumOfElements = numOfElementsForParties;
        if(i<indexForDecreasingSize)
            currentNumOfElements++;

        //fill the send buf according to the number of elements to send to each party
        for(int j=0; j<currentNumOfElements; j++) {

            //xyMinusR[counter] = field->bytesToElement(recBufsBytes[i].data() + (j * fieldByteSize));
            xyMinusR[counter] = recBufsElements[i][j];

            counter++;

        }

    }


    for (int k = 0; k < numOfTrupples; k++)
    {
        cToFill[k] = randomTAnd2TShares[offset + 2*k] + xyMinusR[k];
    }

    offset+=numOfTrupples*2;

}

template<class FieldType>
void ProtocolParty<FieldType>::readclientsinputs(vector<FieldType> &msgsVectorsFlat, vector<FieldType> &squaressVectorsFlat, vector<FieldType> &countersVectorsFlat, vector<FieldType> &unitVectorsFlat){


    vector<FieldType> msg, unitVector;
    FieldType e;

    for(int i=0; i<numClients; i++){

        readServerFile(string(getenv("HOME")) + "/files"+to_string(numClients) + "/server" + to_string(m_partyId) + "ForClient" + to_string(i) + "inputs.bin", msgsVectorsFlat.data() +i*sqrtR*l, squaresVectorsFlat.data() +i*sqrtR*l, countersVectorsFlat.data() +i*sqrtR, unitVectorsFlat.data() + i*sqrtU, &e);
        
        if(i%10000==0) {
            cout<<i<<endl;
        }
    }

}

template<class FieldType>
void ProtocolParty<FieldType>::readServerFile(string fileName, FieldType* msg, FieldType* squares, FieldType* counters, FieldType* unitVector, FieldType * e){

    ifstream inputFile;
    inputFile.open(fileName, std::ios::in | std::ios::binary);

    int msgSize = l*sqrtR;
    int squaresSize = l*sqrtR;
    int countersSize = sqrtR;

    int unitSize = sqrtU;

    inputFile.read((char*)msg, msgSize*4);
    inputFile.read((char*)squares, squaresSize*4);
    inputFile.read((char*)counters, countersSize*4);

    inputFile.read((char*)unitVector, unitSize*4);

    inputFile.read((char*)&e, 4);

    inputFile.close();

}

//
//template <class FieldType>
//int ProtocolParty<FieldType>::validMsgsTest(vector<vector<FieldType>> &msgsVectors, vector<vector<FieldType>> &unitVectors){
//
//    //prepare the random elements for the unit vectors test
//    auto key = generateCommonKey();
//
//    //the number of elements we need to produce for the random bits as well, and thus this depends on the security
//    //parameter and the size of the field. If the security parameter is larger than the field size, we need to generate
//    //more random elements
//    int numOfRandomElements = (sqrtR + l*2 + 1 + sqrtU)*(securityParamter + field->getElementSizeInBits()  + 1)/field->getElementSizeInBits() ;
//
//    auto t1 = high_resolution_clock::now();
//
//    //we use the same rendom elements for all the clients
//    vector<FieldType> randomElements(numOfRandomElements);
//    generatePseudoRandomElements(key, randomElements, randomElements.size());
//    auto t2 = high_resolution_clock::now();
//
//    auto duration = duration_cast<milliseconds>(t2-t1).count();
//    if(flag_print_timings) {
//        cout << "time in milliseconds generatePseudoRandomElements: " << duration << endl;
//    }
//    vector<FieldType> sumXandSqaure(batchSize*l*2);
//
//
//    vector<vector<FieldType>> msgsVectorsForUnitTest(batchSize);
//
//
//    //1. first check that the related index of the second part of a client message is in fact the sqaure of the
//    //related first part of the message
//
//    t1 = high_resolution_clock::now();
//
//
//    int sizeForEachThread;
//    if (batchSize <= numThreads){
//        numThreads = batchSize;
//        sizeForEachThread = 1;
//    } else{
//        sizeForEachThread = (batchSize + numThreads - 1)/ numThreads;
//    }
//
//    cout<<"numThreads is " <<numThreads<<endl;
//    cout<<"num buckets " <<batchSize<<endl;
//
//    //prepareForUnitTest(randomElements, msgsVectors, sumXandSqaure, msgsVectorsForUnitTest);
//
//    for (int t=0; t<numThreads; t++) {
//
//        if ((t + 1) * sizeForEachThread <= batchSize) {
//            threads[t] = thread(&ProtocolParty::prepareForUnitTestFlat, this, ref(randomElements), ref(msgsVectors)  , ref(sumXandSqaure),
//                    ref(msgsVectorsForUnitTest), t * sizeForEachThread, (t + 1) * sizeForEachThread);
//        } else {
//            threads[t] = thread(&ProtocolParty::prepareForUnitTestFlat, this, ref(randomElements), ref(msgsVectors), ref(sumXandSqaure),
//                                ref(msgsVectorsForUnitTest), t * sizeForEachThread, batchSize);
//        }
//    }
//    for (int t=0; t<numThreads; t++){
//        threads[t].join();
//    }
//
////    for(int i=0; i<batchSize; i++){
////
////        msgsVectorsForUnitTest[i].resize(sqrtR, *field->GetZero());
////        for(int k=0; k<sqrtR; k++){
////
////            for(int l1=0; l1< l; l1++){
////
////                //compute the sum of all the elements of the first part for all clients
////                sumXandSqaure[i*l + l1] += msgsVectors[i][l*k + l1];
////
////                //compute the sum of all the elements of the second part for all clients - the squares
////                sumXandSqaure[batchSize*l + i*l + l1] += msgsVectors[i][sqrtR*l+l*k + l1];
////
////                //create the messages for the unit test where each entry of a message is the multiplication of the l-related by a random elements
////                //and summing those with the unit vector with
////                msgsVectorsForUnitTest[i][k] += msgsVectors[i][l*k + l1]*randomElements[sqrtR + l1] +
////                                                msgsVectors[i][sqrtR*l+l*k + l1]*randomElements[sqrtR + l + l1];
////
////
////            }
////
////            //add the share of 0/1 where a share of one should be in the same location of x and x^2 of the message
////            msgsVectorsForUnitTest[i][k] +=  msgsVectors[i][sqrtR*l*2 + k] * randomElements[sqrtR + 2*l];
////
////        }
////
////    }
//
//
//
//    t2 = high_resolution_clock::now();
//
//    duration = duration_cast<milliseconds>(t2-t1).count();
//    if(flag_print_timings) {
//        cout << "time in milliseconds preparing for unit test: " << duration << endl;
//    }
//    //before running the unit test for the compacted message compute the following for every client
//    //1. sumx*sumx
//    //2. sumX *bigR and sumXSqaure *bigR
//
//
//    vector<FieldType> calculatedSqaures(batchSize*l);
//
//    t1 = high_resolution_clock::now();
//
//    cout<< "size of mult in validate is : "<<batchSize*l<<endl;
//
//    DNHonestMultiplication(sumXandSqaure.data(), sumXandSqaure.data(), calculatedSqaures,batchSize*l);
//
//    t2 = high_resolution_clock::now();
//    duration = duration_cast<milliseconds>(t2-t1).count();
//    if(flag_print_timings) {
//        cout << "time in milliseconds DNHonestMultiplication sumXandSqaure: " << duration << endl;
//    }
//    //concatenate the calculated sqares to multiply with bigR
//    sumXandSqaure.insert( sumXandSqaure.end(), calculatedSqaures.begin(), calculatedSqaures.end() );
//
//    //now multiply all these by R
//    //make sure that bigRVec contains enough elements (securityParameter>3*sizeOfMessage)
//    if(bigRVec.size()<3*l*batchSize){ //this will happen at most once
//        int size = bigRVec.size();
//        bigRVec.resize(3*l*batchSize);
//        fill(bigRVec.begin() + size, bigRVec.end(), bigR[0]);
//    }
//    vect
//
//template <class FieldType>
//int ProtocolParty<FieldType>::validMsgsTest(vector<vector<FieldType>> &msgsVectors, vector<vector<FieldType>> &unitVectors){
//
//    //prepare the random elements for the unit vectors test
//    auto key = generateCommonKey();
//

template <class FieldType>
int ProtocolParty<FieldType>::validMsgsTestFlat(vector<FieldType> &msgsVectors, vector<FieldType> &msgsVectorsSquares, vector<FieldType> & counters, vector<FieldType> &unitVectors){

    //prepare the random elements for the unit vectors test
    auto key = generateCommonKey();

    //the number of elements we need to produce for the random bits as well, and thus this depends on the security
    //parameter and the size of the field. If the security parameter is larger than the field size, we need to generate
    //more random elements
    int numOfRandomElements = (sqrtR + l*2 + 1 + sqrtU)*(securityParamter + field->getElementSizeInBits()  + 1)/field->getElementSizeInBits() ;

    auto t1 = high_resolution_clock::now();

    //we use the same rendom elements for all the clients
    vector<FieldType> randomElements(numOfRandomElements);
    generatePseudoRandomElements(key, randomElements, randomElements.size());
    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds generatePseudoRandomElements: " << duration << endl;
    }
    vector<FieldType> sumXandSqaure(batchSize*l*2);


    vector<FieldType> msgsVectorsForUnitTest(batchSize*sqrtR, *field->GetZero());


    //1. first check that the related index of the second part of a client message is in fact the sqaure of the
    //related first part of the message

    t1 = high_resolution_clock::now();


    int sizeForEachThread;
    if (batchSize <= numThreads){
        numThreads = batchSize;
        sizeForEachThread = 1;
    } else{
        sizeForEachThread = (batchSize + numThreads - 1)/ numThreads;
    }
    vector<thread> threads(numThreads);
    cout<<"numThreads is " <<numThreads<<endl;
    cout<<"num buckets " <<batchSize<<endl;

    //prepareForUnitTest(randomElements, msgsVectors, sumXandSqaure, msgsVectorsForUnitTest);

    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * sizeForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::prepareForUnitTestFlat, this, ref(randomElements), ref(msgsVectors),
                    ref(msgsVectorsSquares), ref(counters), ref(sumXandSqaure), ref(msgsVectorsForUnitTest),
                    t * sizeForEachThread, (t + 1) * sizeForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::prepareForUnitTestFlat, this, ref(randomElements), ref(msgsVectors),
                    ref(msgsVectorsSquares), ref(counters), ref(sumXandSqaure), ref(msgsVectorsForUnitTest),
                    t * sizeForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds preparing for unit test: " << duration << endl;
    }
    //before running the unit test for the compacted message compute the following for every client
    //1. sumx*sumx
    //2. sumX *bigR and sumXSqaure *bigR


    vector<FieldType> calculatedSqaures(batchSize*l);

    t1 = high_resolution_clock::now();

    cout<< "size of mult in validate is : "<<batchSize*l<<endl;

    DNHonestMultiplication(sumXandSqaure.data(), sumXandSqaure.data(), calculatedSqaures,batchSize*l);

    t2 = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds DNHonestMultiplication sumXandSqaure: " << duration << endl;
    }
    //concatenate the calculated sqares to multiply with bigR
    sumXandSqaure.insert( sumXandSqaure.end(), calculatedSqaures.begin(), calculatedSqaures.end() );

    //now multiply all these by R
    //make sure that bigRVec contains enough elements (securityParameter>3*sizeOfMessage)
    if(bigRVec.size()<3*l*batchSize){ //this will happen at most once
        int size = bigRVec.size();
        bigRVec.resize(3*l*batchSize);
        fill(bigRVec.begin() + size, bigRVec.end(), bigR[0]);
    }
    vector<FieldType> RTimesSumXandSqaure(sumXandSqaure.size());

    t1 = high_resolution_clock::now();

    cout<< "size of mult in validate is : "<<sumXandSqaure.size()<<endl;


    DNHonestMultiplication(sumXandSqaure.data(), bigRVec.data(), RTimesSumXandSqaure,sumXandSqaure.size());
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds DNHonestMultiplication RTimesSumXandSqaure: " << duration << endl;
    }


    //check the validity of the inputs
    //open v1^2 - v2 and Rv1^2 - Rv2

    //prepare vector for opening
    vector<FieldType> subs(batchSize*l*2);
    vector<FieldType> openedSubs(batchSize*l*2);


    for(int i=0; i<batchSize;i++){

        for(int l1 = 0; l1<l; l1++) {
            subs[i*l + l1] = sumXandSqaure[batchSize*l * 2 + i*l + l1] -
                             sumXandSqaure[batchSize*l + i*l + l1];

            subs[batchSize * l + i*l + l1] = RTimesSumXandSqaure[batchSize*l * 2 + i*l + l1] -
                                             RTimesSumXandSqaure[batchSize*l + i*l + l1];


        }
    }


    int flag = -1;
    openShare(subs.size(), subs, openedSubs, T);

    //now check that all subs are zero
    for(int i=0; i<batchSize; i++){

        for(int l1 = 0; l1 < l; l1++){

            if(openedSubs[i*l + l1] != *field->GetZero() ||
               openedSubs[batchSize*l + i*l + l1] != *field->GetZero()){

                return i;
            }

        }
    }


    t1 = high_resolution_clock::now();


    flag = fasterUnitVectorsTestFlat(msgsVectorsForUnitTest, sqrtR, randomElements.data());

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds unitVectorsTest 1: " << duration << endl;
    }


//    memset((byte*)sum01.data(), 0, 2*batchSize*securityParamter*8);
//    memset((byte*)sum0.data(), 0, batchSize*securityParamter*field->getElementSizeInBytes());
//    memset((byte*)sum1.data(), 0, batchSize*securityParamter*field->getElementSizeInBytes());
    
    cout<<"flag after first unit test is "<<flag<<endl;
//    vector<FieldType> sumOfElementsVecs(batchSize*2, *field->GetZero());
//    vector<FieldType> openedSumOfElementsVecs(batchSize*2, *field->GetZero());

    //flag = -1;//remove after fix
    if(flag==-1) {//all vectors passed the test

        for(int i = 0; i<batchSize; i++) {

            for (int k = 0; k < sqrtR; k++) {

                sumOfElementsVecs[i] += counters[i*sqrtR + k] ;

            }
        }

    }
    else{
        // return flag;
    }

    //open the sums and check that they are equal to 1. This is the counter and should have one in the relevant entry.


    //lastly, check that the unit vectors are indeed unit vector. We can use the same random elements that were already created

    t1 = high_resolution_clock::now();


    flag = fasterUnitVectorsTestFlat(unitVectors, sqrtU, randomElements.data());

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds unitVectorsTest 2: " << duration << endl;
    }

    //do the same check for the unit vectors

    if(flag==-1) {//all vectors passed the test

        for(int i = 0; i<batchSize; i++) {

            for (int k = 0; k < sqrtU; k++) {

                sumOfElementsVecs[i+batchSize] += unitVectors[i*sqrtU + k] ;

            }
        }

    }
    else{
        //return flag;
    }

    //open the sums and check that they are equal to 1. This is the counter and should have one in the relevant entry.

    openShare(sumOfElementsVecs.size(), sumOfElementsVecs, openedSumOfElementsVecs, T);

    for(int i=0; i<batchSize; i++){

        if(openedSumOfElementsVecs[i]!= *field->GetOne()){

            return i;
        }
    }

    return flag;

}


template <class FieldType>
void ProtocolParty<FieldType>::prepareForUnitTestFlat(vector<FieldType> &randomElements, vector<FieldType> &msgsVectors, vector<FieldType> &msgsVectorsSquares, vector<FieldType> & counters,
                                                  vector<FieldType> &sumXandSqaure, vector<FieldType> &msgsVectorsForUnitTest, int start, int end) {
    for(int i=start; i < end; i++){

//        msgsVectorsForUnitTest[i].resize(sqrtR, *field->GetZero());
        for(int k=0; k < sqrtR; k++){

            for(int l1=0; l1 < l; l1++){

                //compute the sum of all the elements of the first part for all clients
                sumXandSqaure[i * l + l1] += msgsVectors[i*sqrtR*l + l * k + l1];

                //compute the sum of all the elements of the second part for all clients - the squares
                sumXandSqaure[batchSize * l + i * l + l1] += msgsVectorsSquares[i*sqrtR * l + l * k + l1];

                //create the messages for the unit test where each entry of a message is the multiplication of the l-related by a random elements
                //and summing those with the unit vector with
                msgsVectorsForUnitTest[i*sqrtR + k] += msgsVectors[i*sqrtR*l + l * k + l1] * randomElements[sqrtR + l1] +
                                                msgsVectors[i*sqrtR * l + l * k + l1] * randomElements[sqrtR + l + l1];


            }

            //add the share of 0/1 where a share of one should be in the same location of x and x^2 of the message
            msgsVectorsForUnitTest[i*sqrtR + k] += counters[i*sqrtR + k] * randomElements[sqrtR + 2 *l];

        }

    }
}

//
//template <class FieldType>
//void ProtocolParty<FieldType>::prepareForUnitTest(vector<FieldType> &randomElements, vector<vector<FieldType>> &msgsVectors,
//                        vector<FieldType> &sumXandSqaure, vector<vector<FieldType>> &msgsVectorsForUnitTest, int start, int end) {
//    for(int i=start; i < end; i++){
//
//        msgsVectorsForUnitTest[i].resize(sqrtR, *field->GetZero());
//        for(int k=0; k < sqrtR; k++){
//
//            for(int l1=0; l1 < l; l1++){
//
//                //compute the sum of all the elements of the first part for all clients
//                sumXandSqaure[i * l + l1] += msgsVectors[i][l * k + l1];
//
//                //compute the sum of all the elements of the second part for all clients - the squares
//                sumXandSqaure[batchSize * l + i * l + l1] += msgsVectors[i][sqrtR * l +
//                                                                                                 l * k + l1];
//
//                //create the messages for the unit test where each entry of a message is the multiplication of the l-related by a random elements
//                //and summing those with the unit vector with
//                msgsVectorsForUnitTest[i][k] += msgsVectors[i][l * k + l1] * randomElements[sqrtR + l1] +
//                                                msgsVectors[i][sqrtR * l + l * k + l1] * randomElements[
//                                                        sqrtR + l + l1];
//
//
//            }
//
//            //add the share of 0/1 where a share of one should be in the same location of x and x^2 of the message
//            msgsVectorsForUnitTest[i][k] += msgsVectors[i][sqrtR * l * 2 + k] * randomElements[sqrtR + 2 *l];
//
//        }
//
//    }
//}



template <class FieldType>
int ProtocolParty<FieldType>::fasterUnitVectorsTestFlat(vector<FieldType> &vecs, int size,
                                                  FieldType *randomElements){
    int flag = -1;

    int sizeForEachThread;
    if (vecs.size() <= numThreads){
        numThreads = vecs.size();
        sizeForEachThread = 1;
    } else{
        sizeForEachThread = (vecs.size() + numThreads - 1)/ numThreads;
    }
    vector<thread> threads(numThreads);

    vector<vector<FieldType>> randomVecs(batchSize, vector<FieldType>(size));
    vector<FieldType> messagesShare(batchSize);
    vector<FieldType> shouldBeZero(batchSize);
    vector<FieldType> openedShouldBeZero(batchSize);
    vector<FieldType> msgByRandomSum(batchSize);
    vector<FieldType> VecByRandomSum(batchSize);

    auto t1 = high_resolution_clock::now();
    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * sizeForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::prepareVecsForUnitTestThreads, this, ref(randomVecs), ref(vecs), ref(messagesShare),
                    ref(msgByRandomSum), ref(VecByRandomSum),randomElements, size, t * sizeForEachThread, (t + 1) * sizeForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::prepareVecsForUnitTestThreads, this, ref(randomVecs), ref(vecs), ref(messagesShare),
                                ref(msgByRandomSum), ref(VecByRandomSum),randomElements, size, t * sizeForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    for(int i=0; i<batchSize; i++){
        shouldBeZero[i] = VecByRandomSum[i]*VecByRandomSum[i] - messagesShare[i]*msgByRandomSum[i];
    }

    //now open shouldBeZero with degree 2d
    openShare(shouldBeZero.size(), shouldBeZero, openedShouldBeZero, 2*T);

    for(int i=0; i<batchSize; i++){

        if(!(openedShouldBeZero[i]==* field->GetZero() ) ){

            flag = i;
            return flag;
        }
    }


    return flag;


}


#ifdef __NVCC__
template <class FieldType>
void ProtocolParty<FieldType>::processSums(FieldType* sum, FieldType* constRandomBits, int size, FieldType* vecs, int device){


    processNN31((merssene31_t *) sum, (merssene31_t *) constRandomBits, size, securityParamter,
                    (merssene31_t *) vecs, batchSize / 8, size, device);

}
#endif

template <class FieldType>
void ProtocolParty<FieldType>::prepareVecsForUnitTestThreads(vector<vector<FieldType>> & randomVecs, vector<FieldType> & vecs, vector<FieldType> & messagesShares,
                                                             vector<FieldType>& msgByRandomSum, vector<FieldType>& VecByRandomSum, FieldType* randomElements, int size, int start, int end){

    for(long i = start; i < end; i++){

        messagesShares[i] = *field->GetZero();
        msgByRandomSum[i] = *field->GetZero();
        VecByRandomSum[i] = *field->GetZero();
        for(long j=0; j<size ; j++){

            randomVecs[i][j] = vecs[i*size + j] * randomElements[j];
            VecByRandomSum[i] += randomVecs[i][j];
            messagesShares[i] += vecs[i*size + j];
            msgByRandomSum[i]+= vecs[i*size + j]*randomElements[j]*randomElements[j];
        }
    }
}



template <class FieldType>
void ProtocolParty<FieldType>::assignSumsPerThreadFlat(vector<long> & sum01, vector<FieldType> & vecs, int size, byte* constRandomBitsPrim,
                                                   vector<vector<FieldType>> & randomVecs, int start, int end){

    for(int i=start; i<end; i++) {
        for (int j = 0; j < securityParamter; j++) {

            for(int k = 0; k<size;k++) {

                //if related bit is zero, accume the sum in sum 0
                sum01[(i*securityParamter + j) + batchSize*securityParamter*constRandomBitsPrim[size* j + k]] +=  randomVecs[i][k].elem;

            }
        }
    }
}

template <class FieldType>
void ProtocolParty<FieldType>::splitShiftFlat(vector<FieldType> &msgsVectors, vector<FieldType> &squaresVectors, vector<FieldType> &countersVectors, vector<FieldType> &unitVectors){


    //generate random shifting for all servers
    vector<int> randomShiftingIndices;
    generateRandomShiftingindices(randomShiftingIndices);

    int shiftRow, shiftCol;

    int numClientsForEachThread;
    if (batchSize <= numThreads){
        numThreads = batchSize;
        numClientsForEachThread = 1;
    } else{
        numClientsForEachThread = (batchSize + numThreads - 1)/ numThreads;
    }
    vector<thread> threads(numThreads);

    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * numClientsForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(msgsVectors), sqrtR, l, 0,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(msgsVectors), sqrtR, l, 0,  t * numClientsForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    int typeSize = field->getElementSizeInBytes();


    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * numClientsForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices),  ref(squaresVectors), sqrtR, l, 0,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(squaresVectors), sqrtR, l, 0,  t * numClientsForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }


    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * numClientsForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(countersVectors), sqrtR, 1, 0,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(countersVectors), sqrtR, 1, 0,  t * numClientsForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }


    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * numClientsForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(unitVectors), sqrtU, 1, 1,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::splitShiftByThreads, this, ref(randomShiftingIndices), ref(unitVectors), sqrtU, 1, 1,  t * numClientsForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    thread ti([&]() {
        if(toUmount) {
            string dir = string(getenv("HOME")) + "/files" + to_string(numClients);
            string command = "umount " + dir;
            int check = system(command.c_str());
            if (!check) {
                cout << "umount succeeded" << endl;
            } else {
                printf("Error : Failed to umount %s\n"
                       "Reason: %s [%d]\n",
                       dir.c_str(), strerror(errno), errno);
            }
            command = "rm -rf " + dir;
            check = system(command.c_str());
            if (!check) {
                cout << "dir termination succeeded" << endl;
            } else {
                printf("Error : Failed to terminate dir %s\n"
                       "Reason: %s [%d]\n",
                       dir.c_str(), strerror(errno), errno);
            }
        }
    });
    ti.detach();


}

template <class FieldType>
void ProtocolParty<FieldType>::splitShiftByThreads(vector<int> & randomShiftingIndices, vector<FieldType> & originalArr, long size, long l, long position, long start, long end){
    long typeSize = field->getElementSizeInBytes();
    long shiftPos;


    vector<FieldType> shiftedArrTemp(size*l);
    for(int i=start; i<end; i++){

        //get the row and col shift
        shiftPos = randomShiftingIndices[2*i + position];//this is for the message , the square and the counter

        //generate the shifted message and assign that back to the msgsVectors
        memcpy(shiftedArrTemp.data() , originalArr.data() + i*size*l+ shiftPos*l, l*(size-shiftPos)*typeSize);
        memcpy(shiftedArrTemp.data() + l*(size-shiftPos), originalArr.data() + i*size*l, shiftPos*l*typeSize);
        memcpy(originalArr.data() + i*size*l, shiftedArrTemp.data() , size*l*typeSize);

    }
}

template <class FieldType>
void ProtocolParty<FieldType>::generateSharedMatricesOptimizedFlat(vector<FieldType> &shiftedMsgsVectors,
                                                              vector<FieldType> &shiftedMsgsVectorsSquares,
                                                              vector<FieldType> &shiftedMsgsVectorsCounters,
                                                              vector<FieldType> &shiftedUnitVectors,
                                                              vector<FieldType> &accMsgsMat,
                                                              vector<FieldType> &accMsgsSquareMat,
                                                              vector<FieldType> &accCountersMat){

    int numOfCols = sqrtR;
    int numOfRows = sqrtU;

    multiplyVectorsWithThreadsFlat(shiftedMsgsVectors, sqrtR*l, shiftedUnitVectors,
            accMsgsMat, numOfRows, numOfCols*l);
    multiplyVectorsWithThreadsFlat(shiftedMsgsVectorsSquares, sqrtR*l, shiftedUnitVectors,
            accMsgsSquareMat, numOfRows, numOfCols*l);
    multiplyVectorsWithThreadsFlat(shiftedMsgsVectorsCounters, sqrtR, shiftedUnitVectors,
            accCountersMat, numOfRows, numOfCols);

}


template <class FieldType>
void ProtocolParty<FieldType>::matrixMulTN(FieldType *C, int ldc, const FieldType *A, int lda, const FieldType *B,
        int ldb, int hA, int wA, int wB)
{

    FieldType sum = *field->GetZero();

	for (int i = 0; i < wA; ++i)
		for (int j = 0; j < wB; ++j)
		{
			sum = *field->GetZero();

			for (int k = 0; k < hA; ++k)
			{
				FieldType a = A[i + k * lda];
				FieldType b = B[j + k * ldb];
				sum += a * b;
			}

			C[i + j * ldc] = sum;
		}
}

template <class FieldType>
void ProtocolParty<FieldType>::regMatrixMulTN(FieldType *C, FieldType *A, int rowa, int cola, FieldType *B,
        int rowb,  int colb)
{

    FieldType sum = *field->GetZero();

    for (int i = 0; i < rowa; i++)
    {
        for (int j = 0; j < colb; j++)
        {
            sum = 0;
            for (int k = 0; k < rowb; k++)
            {
                sum += A[i*cola + k] * B[k*colb + j];
            }

            C[i*colb + j] = sum;
        }
    }
}

template <class FieldType>
void ProtocolParty<FieldType>::multMatricesFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors,
                                            vector<long> & outputDouble, int newNumRows, int newNumCols, int i, __m256i mask, bool toReduce, __m256i & p){

    for(int rowIndex = 0; rowIndex<newNumRows/8; rowIndex++) { //go over each row
        __m256i row = _mm256_maskload_epi32((int *) (unitVectors.data() +i*sqrtU) + rowIndex * 8, mask);

        auto int0 = _mm256_extract_epi32(row, 0);
        auto int1 = _mm256_extract_epi32(row, 1);
        auto int2 = _mm256_extract_epi32(row, 2);
        auto int3 = _mm256_extract_epi32(row, 3);
        auto int4 = _mm256_extract_epi32(row, 4);
        auto int5 = _mm256_extract_epi32(row, 5);
        auto int6 = _mm256_extract_epi32(row, 6);
        auto int7 = _mm256_extract_epi32(row, 7);

        __m256i row0 = _mm256_set1_epi32(int0);
        __m256i row1 = _mm256_set1_epi32(int1);
        __m256i row2 = _mm256_set1_epi32(int2);
        __m256i row3 = _mm256_set1_epi32(int3);
        __m256i row4 = _mm256_set1_epi32(int4);
        __m256i row5 = _mm256_set1_epi32(int5);
        __m256i row6 = _mm256_set1_epi32(int6);
        __m256i row7 = _mm256_set1_epi32(int7);

        for (int colIndex = 0; colIndex < newNumCols / 64; colIndex++) {//go over each message

            //load 8 vectors for 8 small matrices
             auto start = (int *) (input.data() + i*inputSize) + colIndex * 64;

            __m256i col0 = _mm256_maskload_epi32(start, mask);
            __m256i col1 = _mm256_maskload_epi32(start + 8, mask);
            __m256i col2 = _mm256_maskload_epi32(start + 16, mask);
            __m256i col3 = _mm256_maskload_epi32(start + 24, mask);
            __m256i col4 = _mm256_maskload_epi32(start + 32, mask);
            __m256i col5 = _mm256_maskload_epi32(start + 40, mask);
            __m256i col6 = _mm256_maskload_epi32(start + 48, mask);
            __m256i col7 = _mm256_maskload_epi32(start + 56, mask);

            __m256i colHigh0 = _mm256_srli_epi64(col0, 32);
            __m256i colHigh1 = _mm256_srli_epi64(col1, 32);
            __m256i colHigh2 = _mm256_srli_epi64(col2, 32);
            __m256i colHigh3 = _mm256_srli_epi64(col3, 32);
            __m256i colHigh4 = _mm256_srli_epi64(col4, 32);
            __m256i colHigh5 = _mm256_srli_epi64(col5, 32);
            __m256i colHigh6 = _mm256_srli_epi64(col6, 32);
            __m256i colHigh7 = _mm256_srli_epi64(col7, 32);

            //fill each row in every small matrix
            for (int j = 0; j < 8; j++) {

                //load 8 vectors for 8 small matrices
                long *startD =
                        (long *) outputDouble.data() + rowIndex * 8 * newNumCols + j * newNumCols + colIndex * 64;

                __m256i outputLow0 = _mm256_maskload_epi64((long long int *) startD, mask);
                __m256i outputHigh0 = _mm256_maskload_epi64((long long int *) startD + 4, mask);
                __m256i outputLow1 = _mm256_maskload_epi64((long long int *) startD + 8, mask);
                __m256i outputHigh1 = _mm256_maskload_epi64((long long int *) startD + 12, mask);
                __m256i outputLow2 = _mm256_maskload_epi64((long long int *) startD + 16, mask);
                __m256i outputHigh2 = _mm256_maskload_epi64((long long int *) startD + 20, mask);
                __m256i outputLow3 = _mm256_maskload_epi64((long long int *) startD + 24, mask);
                __m256i outputHigh3 = _mm256_maskload_epi64((long long int *) startD + 28, mask);
                __m256i outputLow4 = _mm256_maskload_epi64((long long int *) startD + 32, mask);
                __m256i outputHigh4 = _mm256_maskload_epi64((long long int *) startD + 36, mask);
                __m256i outputLow5 = _mm256_maskload_epi64((long long int *) startD + 40, mask);
                __m256i outputHigh5 = _mm256_maskload_epi64((long long int *) startD + 44, mask);
                __m256i outputLow6 = _mm256_maskload_epi64((long long int *) startD + 48, mask);
                __m256i outputHigh6 = _mm256_maskload_epi64((long long int *) startD + 52, mask);
                __m256i outputLow7 = _mm256_maskload_epi64((long long int *) startD + 56, mask);
                __m256i outputHigh7 = _mm256_maskload_epi64((long long int *) startD + 60, mask);


                __m256i rowI;
                if (j == 0) rowI = row0;
                else if (j == 1) rowI = row1;
                else if (j == 2) rowI = row2;
                else if (j == 3) rowI = row3;
                else if (j == 4) rowI = row4;
                else if (j == 5) rowI = row5;
                else if (j == 6) rowI = row6;
                else if (j == 7) rowI = row7;

                //calc 8 first rows
                __m256i c0 = _mm256_mul_epi32(rowI, col0);
                __m256i c1 = _mm256_mul_epi32(rowI, col1);
                __m256i c2 = _mm256_mul_epi32(rowI, col2);
                __m256i c3 = _mm256_mul_epi32(rowI, col3);
                __m256i c4 = _mm256_mul_epi32(rowI, col4);
                __m256i c5 = _mm256_mul_epi32(rowI, col5);
                __m256i c6 = _mm256_mul_epi32(rowI, col6);
                __m256i c7 = _mm256_mul_epi32(rowI, col7);
                outputLow0 = _mm256_add_epi64(outputLow0, c0);
                outputLow1 = _mm256_add_epi64(outputLow1, c1);
                outputLow2 = _mm256_add_epi64(outputLow2, c2);
                outputLow3 = _mm256_add_epi64(outputLow3, c3);
                outputLow4 = _mm256_add_epi64(outputLow4, c4);
                outputLow5 = _mm256_add_epi64(outputLow5, c5);
                outputLow6 = _mm256_add_epi64(outputLow6, c6);
                outputLow7 = _mm256_add_epi64(outputLow7, c7);

                c0 = _mm256_mul_epi32(rowI, colHigh0);
                c1 = _mm256_mul_epi32(rowI, colHigh1);
                c2 = _mm256_mul_epi32(rowI, colHigh2);
                c3 = _mm256_mul_epi32(rowI, colHigh3);
                c4 = _mm256_mul_epi32(rowI, colHigh4);
                c5 = _mm256_mul_epi32(rowI, colHigh5);
                c6 = _mm256_mul_epi32(rowI, colHigh6);
                c7 = _mm256_mul_epi32(rowI, colHigh7);

                outputHigh0 = _mm256_add_epi64(outputHigh0, c0);
                outputHigh1 = _mm256_add_epi64(outputHigh1, c1);
                outputHigh2 = _mm256_add_epi64(outputHigh2, c2);
                outputHigh3 = _mm256_add_epi64(outputHigh3, c3);
                outputHigh4 = _mm256_add_epi64(outputHigh4, c4);
                outputHigh5 = _mm256_add_epi64(outputHigh5, c5);
                outputHigh6 = _mm256_add_epi64(outputHigh6, c6);
                outputHigh7 = _mm256_add_epi64(outputHigh7, c7);

                if (toReduce){
                    reduce(outputLow0, outputLow1, outputLow2, outputLow3, outputLow4,
                           outputLow5, outputLow6, outputLow7, p);
                    reduce(outputHigh0, outputHigh1, outputHigh2, outputHigh3, outputHigh4,
                           outputHigh5, outputHigh6, outputHigh7, p);
                }

                _mm256_maskstore_epi64((long long int *) startD, mask, outputLow0);
                _mm256_maskstore_epi64((long long int *) startD + 4, mask, outputHigh0);
                _mm256_maskstore_epi64((long long int *) startD + 8, mask, outputLow1);
                _mm256_maskstore_epi64((long long int *) startD + 12, mask, outputHigh1);
                _mm256_maskstore_epi64((long long int *) startD + 16, mask, outputLow2);
                _mm256_maskstore_epi64((long long int *) startD + 20, mask, outputHigh2);
                _mm256_maskstore_epi64((long long int *) startD + 24, mask, outputLow3);
                _mm256_maskstore_epi64((long long int *) startD + 28, mask, outputHigh3);
                _mm256_maskstore_epi64((long long int *) startD + 32, mask, outputLow4);
                _mm256_maskstore_epi64((long long int *) startD + 36, mask, outputHigh4);
                _mm256_maskstore_epi64((long long int *) startD + 40, mask, outputLow5);
                _mm256_maskstore_epi64((long long int *) startD + 44, mask, outputHigh5);
                _mm256_maskstore_epi64((long long int *) startD + 48, mask, outputLow6);
                _mm256_maskstore_epi64((long long int *) startD + 52, mask, outputHigh6);
                _mm256_maskstore_epi64((long long int *) startD + 56, mask, outputLow7);
                _mm256_maskstore_epi64((long long int *) startD + 60, mask, outputHigh7);


            }

        }
    }

}

template <class FieldType>
void ProtocolParty<FieldType>::reduce(__m256i & output0, __m256i & output1, __m256i & output2, __m256i & output3,
                                      __m256i & output4, __m256i & output5, __m256i & output6, __m256i & output7,
                                      __m256i & p) {
//get the bottom 31 bit
    __m256i bottom0 = _mm256_and_si256(output0, p);
    __m256i bottom1 = _mm256_and_si256(output1, p);
    __m256i bottom2 = _mm256_and_si256(output2, p);
    __m256i bottom3 = _mm256_and_si256(output3, p);
    __m256i bottom4 = _mm256_and_si256(output4, p);
    __m256i bottom5 = _mm256_and_si256(output5, p);
    __m256i bottom6 = _mm256_and_si256(output6, p);
    __m256i bottom7 = _mm256_and_si256(output7, p);
//                    unsigned int bottom = multLong & p;

    //get the top 31 bits

    __m256i top0 = _mm256_srli_epi64(output0, 31);
    __m256i top1 = _mm256_srli_epi64(output1, 31);
    __m256i top2 = _mm256_srli_epi64(output2, 31);
    __m256i top3 = _mm256_srli_epi64(output3, 31);
    __m256i top4 = _mm256_srli_epi64(output4, 31);
    __m256i top5 = _mm256_srli_epi64(output5, 31);
    __m256i top6 = _mm256_srli_epi64(output6, 31);
    __m256i top7 = _mm256_srli_epi64(output7, 31);


    top0 = _mm256_and_si256(top0, p);
    top1 = _mm256_and_si256(top1, p);
    top2 = _mm256_and_si256(top2, p);
    top3 = _mm256_and_si256(top3, p);
    top4 = _mm256_and_si256(top4, p);
    top5 = _mm256_and_si256(top5, p);
    top6 = _mm256_and_si256(top6, p);
    top7 = _mm256_and_si256(top7, p);
//                    unsigned int top = (multLong>>31);

    __m256i res0 = _mm256_add_epi64(bottom0, top0);
    __m256i res1 = _mm256_add_epi64(bottom1, top1);
    __m256i res2 = _mm256_add_epi64(bottom2, top2);
    __m256i res3 = _mm256_add_epi64(bottom3, top3);
    __m256i res4 = _mm256_add_epi64(bottom4, top4);
    __m256i res5 = _mm256_add_epi64(bottom5, top5);
    __m256i res6 = _mm256_add_epi64(bottom6, top6);
    __m256i res7 = _mm256_add_epi64(bottom7, top7);


    top0 = _mm256_srli_epi64(output0, 62);
    top1 = _mm256_srli_epi64(output1, 62);
    top2 = _mm256_srli_epi64(output2, 62);
    top3 = _mm256_srli_epi64(output3, 62);
    top4 = _mm256_srli_epi64(output4, 62);
    top5 = _mm256_srli_epi64(output5, 62);
    top6 = _mm256_srli_epi64(output6, 62);
    top7 = _mm256_srli_epi64(output7, 62);


    top0 = _mm256_and_si256(top0, p);
    top1 = _mm256_and_si256(top1, p);
    top2 = _mm256_and_si256(top2, p);
    top3 = _mm256_and_si256(top3, p);
    top4 = _mm256_and_si256(top4, p);
    top5 = _mm256_and_si256(top5, p);
    top6 = _mm256_and_si256(top6, p);
    top7 = _mm256_and_si256(top7, p);
//                    unsigned int top = (multLong>>31);

    res0 = _mm256_add_epi64(res0, top0);
    res1 = _mm256_add_epi64(res1, top1);
    res2 = _mm256_add_epi64(res2, top2);
    res3 = _mm256_add_epi64(res3, top3);
    res4 = _mm256_add_epi64(res4, top4);
    res5 = _mm256_add_epi64(res5, top5);
    res6 = _mm256_add_epi64(res6, top6);
    res7 = _mm256_add_epi64(res7, top7);

    //maximim the value of 2p-2
//                    if(answer.elem>=p)
//                        answer.elem-=p;
//
////                    }
    __m256i test0 = _mm256_cmpgt_epi32(res0, p);
    __m256i test1 = _mm256_cmpgt_epi32(res1, p);
    __m256i test2 = _mm256_cmpgt_epi32(res2, p);
    __m256i test3 = _mm256_cmpgt_epi32(res3, p);
    __m256i test4 = _mm256_cmpgt_epi32(res4, p);
    __m256i test5 = _mm256_cmpgt_epi32(res5, p);
    __m256i test6 = _mm256_cmpgt_epi32(res6, p);
    __m256i test7 = _mm256_cmpgt_epi32(res7, p);

    __m256i sub0 = _mm256_and_si256(test0, p);
    __m256i sub1 = _mm256_and_si256(test1, p);
    __m256i sub2 = _mm256_and_si256(test2, p);
    __m256i sub3 = _mm256_and_si256(test3, p);
    __m256i sub4 = _mm256_and_si256(test4, p);
    __m256i sub5 = _mm256_and_si256(test5, p);
    __m256i sub6 = _mm256_and_si256(test6, p);
    __m256i sub7 = _mm256_and_si256(test7, p);

    output0 = _mm256_sub_epi32(res0, sub0);
    output1 = _mm256_sub_epi32(res1, sub1);
    output2 = _mm256_sub_epi32(res2, sub2);
    output3 = _mm256_sub_epi32(res3, sub3);
    output4 = _mm256_sub_epi32(res4, sub4);
    output5 = _mm256_sub_epi32(res5, sub5);
    output6 = _mm256_sub_epi32(res6, sub6);
    output7 = _mm256_sub_epi32(res7, sub7);
}


template <class FieldType>
void ProtocolParty<FieldType>::reduceMatrix(vector<long> & outputDouble, int newNumRows, int newNumCols,
        __m256i mask, __m256i p){

    for(int rowIndex = 0; rowIndex<newNumRows; rowIndex++){ //go over each row

        for(int colIndex=0; colIndex<newNumCols / 32; colIndex++){//go over each message

            //load 8 vectors for 8 small matrices
            auto startD = (long*) outputDouble.data() + rowIndex * newNumCols + colIndex * 32;

            __m256i output0 = _mm256_maskload_epi64((long long int*)startD, mask);
            __m256i output1 = _mm256_maskload_epi64((long long int*)startD + 4, mask);
            __m256i output2 = _mm256_maskload_epi64((long long int*)startD + 8, mask);
            __m256i output3 = _mm256_maskload_epi64((long long int*)startD + 12, mask);
            __m256i output4 = _mm256_maskload_epi64((long long int*)startD + 16, mask);
            __m256i output5 = _mm256_maskload_epi64((long long int*)startD + 20, mask);
            __m256i output6 = _mm256_maskload_epi64((long long int*)startD + 24, mask);
            __m256i output7 = _mm256_maskload_epi64((long long int*)startD + 28, mask);


            //get the bottom 31 bit
            __m256i bottom0 = _mm256_and_si256(output0, p);
            __m256i bottom1 = _mm256_and_si256(output1, p);
            __m256i bottom2 = _mm256_and_si256(output2, p);
            __m256i bottom3 = _mm256_and_si256(output3, p);
            __m256i bottom4 = _mm256_and_si256(output4, p);
            __m256i bottom5 = _mm256_and_si256(output5, p);
            __m256i bottom6 = _mm256_and_si256(output6, p);
            __m256i bottom7 = _mm256_and_si256(output7, p);
//                    unsigned int bottom = multLong & p;

            //get the top 31 bits

            __m256i top0 = _mm256_srli_epi64(output0, 31);
            __m256i top1 = _mm256_srli_epi64(output1, 31);
            __m256i top2 = _mm256_srli_epi64(output2, 31);
            __m256i top3 = _mm256_srli_epi64(output3, 31);
            __m256i top4 = _mm256_srli_epi64(output4, 31);
            __m256i top5 = _mm256_srli_epi64(output5, 31);
            __m256i top6 = _mm256_srli_epi64(output6, 31);
            __m256i top7 = _mm256_srli_epi64(output7, 31);


            top0 = _mm256_and_si256(top0, p);
            top1 = _mm256_and_si256(top1, p);
            top2 = _mm256_and_si256(top2, p);
            top3 = _mm256_and_si256(top3, p);
            top4 = _mm256_and_si256(top4, p);
            top5 = _mm256_and_si256(top5, p);
            top6 = _mm256_and_si256(top6, p);
            top7 = _mm256_and_si256(top7, p);
//                    unsigned int top = (multLong>>31);

            __m256i res0 = _mm256_add_epi64(bottom0, top0);
            __m256i res1 = _mm256_add_epi64(bottom1, top1);
            __m256i res2 = _mm256_add_epi64(bottom2, top2);
            __m256i res3 = _mm256_add_epi64(bottom3, top3);
            __m256i res4 = _mm256_add_epi64(bottom4, top4);
            __m256i res5 = _mm256_add_epi64(bottom5, top5);
            __m256i res6 = _mm256_add_epi64(bottom6, top6);
            __m256i res7 = _mm256_add_epi64(bottom7, top7);


            top0 = _mm256_srli_epi64(output0, 62);
            top1 = _mm256_srli_epi64(output1, 62);
            top2 = _mm256_srli_epi64(output2, 62);
            top3 = _mm256_srli_epi64(output3, 62);
            top4 = _mm256_srli_epi64(output4, 62);
            top5 = _mm256_srli_epi64(output5, 62);
            top6 = _mm256_srli_epi64(output6, 62);
            top7 = _mm256_srli_epi64(output7, 62);


            top0 = _mm256_and_si256(top0, p);
            top1 = _mm256_and_si256(top1, p);
            top2 = _mm256_and_si256(top2, p);
            top3 = _mm256_and_si256(top3, p);
            top4 = _mm256_and_si256(top4, p);
            top5 = _mm256_and_si256(top5, p);
            top6 = _mm256_and_si256(top6, p);
            top7 = _mm256_and_si256(top7, p);
//                    unsigned int top = (multLong>>31);

            res0 = _mm256_add_epi64(res0, top0);
            res1 = _mm256_add_epi64(res1, top1);
            res2 = _mm256_add_epi64(res2, top2);
            res3 = _mm256_add_epi64(res3, top3);
            res4 = _mm256_add_epi64(res4, top4);
            res5 = _mm256_add_epi64(res5, top5);
            res6 = _mm256_add_epi64(res6, top6);
            res7 = _mm256_add_epi64(res7, top7);

            //maximim the value of 2p-2
//                    if(answer.elem>=p)
//                        answer.elem-=p;
//
////                    }
            __m256i test0 = _mm256_cmpgt_epi32(res0, p);
            __m256i test1 = _mm256_cmpgt_epi32(res1, p);
            __m256i test2 = _mm256_cmpgt_epi32(res2, p);
            __m256i test3 = _mm256_cmpgt_epi32(res3, p);
            __m256i test4 = _mm256_cmpgt_epi32(res4, p);
            __m256i test5 = _mm256_cmpgt_epi32(res5, p);
            __m256i test6 = _mm256_cmpgt_epi32(res6, p);
            __m256i test7 = _mm256_cmpgt_epi32(res7, p);

            __m256i sub0 = _mm256_and_si256(test0, p);
            __m256i sub1 = _mm256_and_si256(test1, p);
            __m256i sub2 = _mm256_and_si256(test2, p);
            __m256i sub3 = _mm256_and_si256(test3, p);
            __m256i sub4 = _mm256_and_si256(test4, p);
            __m256i sub5 = _mm256_and_si256(test5, p);
            __m256i sub6 = _mm256_and_si256(test6, p);
            __m256i sub7 = _mm256_and_si256(test7, p);

            res0 = _mm256_sub_epi32(res0, sub0);
            res1 = _mm256_sub_epi32(res1, sub1);
            res2 = _mm256_sub_epi32(res2, sub2);
            res3 = _mm256_sub_epi32(res3, sub3);
            res4 = _mm256_sub_epi32(res4, sub4);
            res5 = _mm256_sub_epi32(res5, sub5);
            res6 = _mm256_sub_epi32(res6, sub6);
            res7 = _mm256_sub_epi32(res7, sub7);


            _mm256_maskstore_epi64((long long int*)startD, mask, res0);
            _mm256_maskstore_epi64((long long int*)startD + 4, mask, res1);
            _mm256_maskstore_epi64((long long int*)startD + 8, mask, res2);
            _mm256_maskstore_epi64((long long int*)startD + 12, mask, res3);
            _mm256_maskstore_epi64((long long int*)startD + 16, mask, res4);
            _mm256_maskstore_epi64((long long int*)startD + 20, mask, res5);
            _mm256_maskstore_epi64((long long int*)startD + 24, mask, res6);
            _mm256_maskstore_epi64((long long int*)startD + 28, mask, res7);

//                    answer.elem = bottom + top;

        }
    }
}
//
//template <class FieldType>
//void ProtocolParty<FieldType>::multiplyVectorsWithThreads(vector<vector<FieldType>> & input, vector<vector<FieldType>> & unitVectors,
//                                                vector<FieldType> & output, int numOfRows, int numOfCols){
//
//    int toReduce = 0; //Every 4 multiplications there is need to reduce all table
//    __m256i mask = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
//    __m256i p = _mm256_set_epi32(0, 2147483647, 0, 2147483647, 0, 2147483647, 0, 2147483647);
//
//    int newNumRows = numOfRows;
//    if (numOfRows % 8 != 0) {
//        newNumRows = (numOfRows / 8)*8 + 8;
//        for( int i=0; i<unitVectors.size(); i++) {
//            unitVectors[i].resize(newNumRows);
//        }
//    }
//
//    int newNumCols = numOfCols;
//    if (numOfCols % 64 != 0) {
//        newNumCols = (numOfCols / 64)*64 + 64;
//        for( int i=0; i<input.size(); i++){
//            input[i].resize(newNumCols);
//        }
//    }
//
//    int numClientsForEachThread;
//    if (input.size() <= numThreads){
//        numThreads = input.size();
//        numClientsForEachThread = 1;
//    } else{
//        numClientsForEachThread = (input.size() + numThreads - 1)/ numThreads;
//    }
//
//    vector<vector<long>> outputDoublePerThread(numThreads, vector<long>(newNumRows*newNumCols));
//    vector<long> outputDouble(newNumRows*newNumCols);
//
//    for (int t=0; t<numThreads; t++) {
//
//            if ((t + 1) * numClientsForEachThread <= input.size()) {
//                threads[t] = thread(&ProtocolParty::multiplyVectorsPerThread, this, ref(input), ref(unitVectors), ref(outputDoublePerThread[t]), newNumRows, newNumCols,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
//            } else {
//                threads[t] = thread(&ProtocolParty::multiplyVectorsPerThread, this, ref(input), ref(unitVectors), ref(outputDoublePerThread[t]), newNumRows, newNumCols,  t * numClientsForEachThread, input.size());
//            }
//        }
//        for (int t=0; t<numThreads; t++){
//            threads[t].join();
//        }
//
//
//    for(int t=0; t<numThreads; t++){//go over each client
//
//        for(int rowIndex = 0; rowIndex<newNumRows; rowIndex++){ //go over each row
//
//            for(int colIndex=0; colIndex<newNumCols / 32; colIndex++) {//go over each message
//
//                //load 8 vectors for 8 small matrices
//                auto threadRow = (long *) outputDoublePerThread[t].data() + rowIndex * newNumCols + colIndex * 32;
//                auto outputRow = (long *) outputDouble.data() + rowIndex * newNumCols + colIndex * 32;
//
//                __m256i threadRow0 = _mm256_maskload_epi64((long long int *) threadRow, mask);
//                __m256i threadRow1 = _mm256_maskload_epi64((long long int *) threadRow + 4, mask);
//                __m256i threadRow2 = _mm256_maskload_epi64((long long int *) threadRow + 8, mask);
//                __m256i threadRow3 = _mm256_maskload_epi64((long long int *) threadRow + 12, mask);
//                __m256i threadRow4 = _mm256_maskload_epi64((long long int *) threadRow + 16, mask);
//                __m256i threadRow5 = _mm256_maskload_epi64((long long int *) threadRow + 20, mask);
//                __m256i threadRow6 = _mm256_maskload_epi64((long long int *) threadRow + 24, mask);
//                __m256i threadRow7 = _mm256_maskload_epi64((long long int *) threadRow + 28, mask);
//
//
//                __m256i outputRow0 = _mm256_maskload_epi64((long long int *) outputRow, mask);
//                __m256i outputRow1 = _mm256_maskload_epi64((long long int *) outputRow + 4, mask);
//                __m256i outputRow2 = _mm256_maskload_epi64((long long int *) outputRow + 8, mask);
//                __m256i outputRow3 = _mm256_maskload_epi64((long long int *) outputRow + 12, mask);
//                __m256i outputRow4 = _mm256_maskload_epi64((long long int *) outputRow + 16, mask);
//                __m256i outputRow5 = _mm256_maskload_epi64((long long int *) outputRow + 20, mask);
//                __m256i outputRow6 = _mm256_maskload_epi64((long long int *) outputRow + 24, mask);
//                __m256i outputRow7 = _mm256_maskload_epi64((long long int *) outputRow + 28, mask);
//
//                outputRow0 = _mm256_add_epi64(threadRow0, outputRow0);
//                outputRow1 = _mm256_add_epi64(threadRow1, outputRow1);
//                outputRow2 = _mm256_add_epi64(threadRow2, outputRow2);
//                outputRow3 = _mm256_add_epi64(threadRow3, outputRow3);
//                outputRow4 = _mm256_add_epi64(threadRow4, outputRow4);
//                outputRow5 = _mm256_add_epi64(threadRow5, outputRow5);
//                outputRow6 = _mm256_add_epi64(threadRow6, outputRow6);
//                outputRow7 = _mm256_add_epi64(threadRow7, outputRow7);
//
//                _mm256_maskstore_epi64((long long int *) outputRow, mask, outputRow0);
//                _mm256_maskstore_epi64((long long int *) outputRow + 4, mask, outputRow1);
//                _mm256_maskstore_epi64((long long int *) outputRow + 8, mask, outputRow2);
//                _mm256_maskstore_epi64((long long int *) outputRow + 12, mask, outputRow3);
//                _mm256_maskstore_epi64((long long int *) outputRow + 16, mask, outputRow4);
//                _mm256_maskstore_epi64((long long int *) outputRow + 20, mask, outputRow5);
//                _mm256_maskstore_epi64((long long int *) outputRow + 24, mask, outputRow6);
//                _mm256_maskstore_epi64((long long int *) outputRow + 28, mask, outputRow7);
//
//            }
//        }
//
//        toReduce ++;
//
//        if (toReduce == 32 || t == numThreads - 1){
////            //reduce all matrix
//            reduceMatrix(outputDouble, newNumRows, newNumCols, mask, p);
//            toReduce = 0;
//        }
//
//    }
////    cout<<"output double:"<<endl;
////    for(int rowIndex = 0; rowIndex<newNumRows; rowIndex++) { //go over each row
////        for (int colIndex = 0; colIndex < newNumCols; colIndex++) {//go over each message
////            cout<<outputDouble[rowIndex * newNumCols  + colIndex] << " ";
////        }
////        cout<<endl;
////    }
//
//    auto t1 = high_resolution_clock::now();
//    int numBlocks = (numOfCols % 8 == 0) ? numOfCols/8 : numOfCols/8 + 1;
//    int remain = (numOfCols % 8 == 0) ? 8 : numOfCols%8;
//
//    for(int rowIndex = 0; rowIndex<numOfRows; rowIndex++){ //go over each row
//
//        for(int colIndex=0; colIndex<numBlocks; colIndex++){//go over each message
//
//            for (int k=0; (colIndex < numBlocks-1 && k<8) || (colIndex == numBlocks-1 && k<remain); k++) {
//
//                if (k % 2 != 0) {
//                    //get the high int
//                    output[rowIndex * numOfCols + colIndex * 8 + k] = (int)outputDouble[rowIndex * newNumCols + colIndex * 8 + 4 + k/2];
//                } else {
//                    //get the low int
//                    output[rowIndex * numOfCols + colIndex * 8 + k] = (int)outputDouble[rowIndex * newNumCols  + colIndex * 8 + k/2];
//                }
//            }
//        }
//    }
//
//    auto t2 = high_resolution_clock::now();
//
//    auto duration = duration_cast<milliseconds>(t2-t1).count();
//    if(flag_print_timings) {
//        cout << "time in milliseconds copy output: " << duration << endl;
//    }
////    cout<<"output:"<<endl;
////    for(int rowIndex = 0; rowIndex<numOfRows; rowIndex++) { //go over each row
////
////        for (int colIndex = 0; colIndex < numOfCols; colIndex++) {//go over each message
////            cout<<output[rowIndex * numOfCols  + colIndex] << " ";
////        }
////        cout<<endl;
////    }
//}


template <class FieldType>
void ProtocolParty<FieldType>::multiplyVectorsWithThreadsFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors,
                                                          vector<FieldType> & output, int numOfRows, int numOfCols){

    int toReduce = 0; //Every 4 multiplications there is need to reduce all table
    __m256i mask = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    __m256i p = _mm256_set_epi32(0, 2147483647, 0, 2147483647, 0, 2147483647, 0, 2147483647);

    int newNumRows = numOfRows;
    if (numOfRows % 8 != 0) {
        newNumRows = (numOfRows / 8)*8 + 8;
//        for( int i=0; i<batchSize; i++) {
//            unitVectors[i].resize(newNumRows);
//        }
    }

    int newNumCols = numOfCols;
    if (numOfCols % 64 != 0) {
        newNumCols = (numOfCols / 64)*64 + 64;
//        for( int i=0; i<batchSize; i++){
//            input[i].resize(newNumCols);
//        }
    }


    auto start = high_resolution_clock::now();
//numThreads = 4;
    int numClientsForEachThread;
    if (batchSize <= numThreads){
        numThreads = batchSize;
        numClientsForEachThread = 1;
    } else{
        numClientsForEachThread = (batchSize + numThreads - 1)/ numThreads;
    }
    vector<thread> threads(numThreads);

    vector<vector<long>> outputDoublePerThread(numThreads, vector<long>(newNumRows*newNumCols));
    vector<long> outputDouble(newNumRows*newNumCols);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end-start).count();
    cout << "create matrices took: " << duration << endl;

    start = high_resolution_clock::now();

    for (int t=0; t<numThreads; t++) {

        if ((t + 1) * numClientsForEachThread <= batchSize) {
            threads[t] = thread(&ProtocolParty::multiplyVectorsPerThreadFlat, this, ref(input), inputSize, ref(unitVectors), ref(outputDoublePerThread[t]), newNumRows, newNumCols,  t * numClientsForEachThread, (t + 1) * numClientsForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::multiplyVectorsPerThreadFlat, this, ref(input), inputSize, ref(unitVectors), ref(outputDoublePerThread[t]), newNumRows, newNumCols,  t * numClientsForEachThread, batchSize);
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end-start).count();
    cout << "all threads work took: " << duration << endl;

    start = high_resolution_clock::now();
    for(int t=0; t<numThreads; t++){//go over each client

        for(int rowIndex = 0; rowIndex<newNumRows; rowIndex++){ //go over each row

            for(int colIndex=0; colIndex<newNumCols / 32; colIndex++) {//go over each message

                //load 8 vectors for 8 small matrices
                auto threadRow = (long *) outputDoublePerThread[t].data() + rowIndex * newNumCols + colIndex * 32;
                auto outputRow = (long *) outputDouble.data() + rowIndex * newNumCols + colIndex * 32;

                __m256i threadRow0 = _mm256_maskload_epi64((long long int *) threadRow, mask);
                __m256i threadRow1 = _mm256_maskload_epi64((long long int *) threadRow + 4, mask);
                __m256i threadRow2 = _mm256_maskload_epi64((long long int *) threadRow + 8, mask);
                __m256i threadRow3 = _mm256_maskload_epi64((long long int *) threadRow + 12, mask);
                __m256i threadRow4 = _mm256_maskload_epi64((long long int *) threadRow + 16, mask);
                __m256i threadRow5 = _mm256_maskload_epi64((long long int *) threadRow + 20, mask);
                __m256i threadRow6 = _mm256_maskload_epi64((long long int *) threadRow + 24, mask);
                __m256i threadRow7 = _mm256_maskload_epi64((long long int *) threadRow + 28, mask);


                __m256i outputRow0 = _mm256_maskload_epi64((long long int *) outputRow, mask);
                __m256i outputRow1 = _mm256_maskload_epi64((long long int *) outputRow + 4, mask);
                __m256i outputRow2 = _mm256_maskload_epi64((long long int *) outputRow + 8, mask);
                __m256i outputRow3 = _mm256_maskload_epi64((long long int *) outputRow + 12, mask);
                __m256i outputRow4 = _mm256_maskload_epi64((long long int *) outputRow + 16, mask);
                __m256i outputRow5 = _mm256_maskload_epi64((long long int *) outputRow + 20, mask);
                __m256i outputRow6 = _mm256_maskload_epi64((long long int *) outputRow + 24, mask);
                __m256i outputRow7 = _mm256_maskload_epi64((long long int *) outputRow + 28, mask);

                outputRow0 = _mm256_add_epi64(threadRow0, outputRow0);
                outputRow1 = _mm256_add_epi64(threadRow1, outputRow1);
                outputRow2 = _mm256_add_epi64(threadRow2, outputRow2);
                outputRow3 = _mm256_add_epi64(threadRow3, outputRow3);
                outputRow4 = _mm256_add_epi64(threadRow4, outputRow4);
                outputRow5 = _mm256_add_epi64(threadRow5, outputRow5);
                outputRow6 = _mm256_add_epi64(threadRow6, outputRow6);
                outputRow7 = _mm256_add_epi64(threadRow7, outputRow7);

                _mm256_maskstore_epi64((long long int *) outputRow, mask, outputRow0);
                _mm256_maskstore_epi64((long long int *) outputRow + 4, mask, outputRow1);
                _mm256_maskstore_epi64((long long int *) outputRow + 8, mask, outputRow2);
                _mm256_maskstore_epi64((long long int *) outputRow + 12, mask, outputRow3);
                _mm256_maskstore_epi64((long long int *) outputRow + 16, mask, outputRow4);
                _mm256_maskstore_epi64((long long int *) outputRow + 20, mask, outputRow5);
                _mm256_maskstore_epi64((long long int *) outputRow + 24, mask, outputRow6);
                _mm256_maskstore_epi64((long long int *) outputRow + 28, mask, outputRow7);

            }
        }

        toReduce ++;

//        if (toReduce == 32 || t == numThreads - 1){
//            cout<<"in reduce"<<endl;
//            //reduce all matrix
            reduceMatrix((vector<long>&)outputDouble, newNumRows, newNumCols, mask, p);
            toReduce = 0;
//        }

    }

    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end-start).count();
    cout << "reduce took: " << duration << endl;
//    cout<<"output double:"<<endl;
//    for(int rowIndex = 0; rowIndex<newNumRows; rowIndex++) { //go over each row
//        for (int colIndex = 0; colIndex < newNumCols; colIndex++) {//go over each message
//            cout<<outputDouble[rowIndex * newNumCols  + colIndex] << " ";
//        }
//        cout<<endl;
//    }

    auto t1 = high_resolution_clock::now();
    int numBlocks = (numOfCols % 8 == 0) ? numOfCols/8 : numOfCols/8 + 1;
    int remain = (numOfCols % 8 == 0) ? 8 : numOfCols%8;

    for(int rowIndex = 0; rowIndex<numOfRows; rowIndex++){ //go over each row

        for(int colIndex=0; colIndex<numBlocks; colIndex++){//go over each message

            for (int k=0; (colIndex < numBlocks-1 && k<8) || (colIndex == numBlocks-1 && k<remain); k++) {

                if (k % 2 != 0) {
                    //get the high int
                    output[rowIndex * numOfCols + colIndex * 8 + k] = (int)outputDouble[rowIndex * newNumCols + colIndex * 8 + 4 + k/2];
                } else {
                    //get the low int
                    output[rowIndex * numOfCols + colIndex * 8 + k] = (int)outputDouble[rowIndex * newNumCols  + colIndex * 8 + k/2];
                }
            }
        }
    }

    auto t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds copy output: " << duration << endl;
    }
//    cout<<"output:"<<endl;
//    for(int rowIndex = 0; rowIndex<numOfRows; rowIndex++) { //go over each row
//
//        for (int colIndex = 0; colIndex < numOfCols; colIndex++) {//go over each message
//            cout<<output[rowIndex * numOfCols  + colIndex] << " ";
//        }
//        cout<<endl;
//    }
//numThreads=4;
}
//
//template <class FieldType>
//void ProtocolParty<FieldType>::multiplyVectorsPerThread(vector<vector<FieldType>> & input, vector<vector<FieldType>> & unitVectors, vector<long> & outputDouble,
//                                                        int newNumRows, int newNumCols, int start, int end){
//
//    __m256i mask = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
//    __m256i p = _mm256_set_epi32(0, 2147483647, 0, 2147483647, 0, 2147483647, 0, 2147483647);
//    int toReduce = 0;
//
//    for(int i=start; i<end; i++){//go over each client
//
//        multMatrices(input, unitVectors, outputDouble, newNumRows, newNumCols, i, mask);
//
//        toReduce += 2;
//
//        if (toReduce == 4 || i == input.size()-1){
////            //reduce all matrix
//            reduceMatrix(outputDouble, newNumRows, newNumCols, mask, p);
//
//            toReduce = 0;
//        }
//
//    }
//}


template <class FieldType>
void ProtocolParty<FieldType>::multiplyVectorsPerThreadFlat(vector<FieldType> & input, int inputSize, vector<FieldType> & unitVectors, vector<long> & outputDouble,
                                                        int newNumRows, int newNumCols, int start, int end){

    __m256i mask = _mm256_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    __m256i p = _mm256_set_epi32(0, 2147483647, 0, 2147483647, 0, 2147483647, 0, 2147483647);
//    int toReduce = 0;
    bool toReduce = false;
    for(int i=start; i<end; i++){//go over each client
        if (i == batchSize - 1) toReduce = true;
        multMatricesFlat(input, inputSize, unitVectors, outputDouble, newNumRows, newNumCols, i, mask, toReduce, p);

        toReduce = !toReduce;

//        if (toReduce == 4 || i == batchSize - 1){
////            //reduce all matrix
//            reducestart = high_resolution_clock::now();
//            reduceMatrix(outputDouble, newNumRows, newNumCols, mask, p);
//            reduceend = high_resolution_clock::now();
//            reduceduration += duration_cast<nanoseconds>(reduceend-reducestart).count();
//            toReduce = 0;
//        }

    }

}

template <class FieldType>
void ProtocolParty<FieldType>::generateRandomShiftingindices(vector<int> &randomShiftingVec){

    randomShiftingVec.resize(2*numClients);

    //prepare the random elements for the unit vectors test
    auto key = generateCommonKey();


    //we use the same rendom elements for all the clients. A size of an int is enough and thus we use at most 4 bytes
    vector<FieldType> randomElements((numClients*2 *4)/field->getElementSizeInBytes());
    generatePseudoRandomElements(key, randomElements, randomElements.size());

    int *randomInts = (int *)randomElements.data();

    //go over each element and get the random position

    for(int i=0; i<numClients; i++){

        randomShiftingVec[2*i] = abs(randomInts[i]) % sqrtR;
        randomShiftingVec[2*i+1] = abs(randomInts[i]) % sqrtU;
    }

}
template <class FieldType>
void ProtocolParty<FieldType>::commitOnMatrices(vector<FieldType> &accMats, vector<FieldType> &accFieldCountersMat,
                                               vector<vector<byte>> &recBufsBytes){


    OpenSSLSHA256 hash;
    vector<byte> byteVec(accMats.size()*field->getElementSizeInBytes());

    field->elementVectorToByteVector(accMats, byteVec);

    //hash the messsages and the counters shares
    hash.update(byteVec, 0, byteVec.size());
    vector<byte> out;
    hash.hashFinal(out, 0);

    byteVec.resize(accFieldCountersMat.size()*field->getElementSizeInBytes());

    field->elementVectorToByteVector(accFieldCountersMat, byteVec);

    hash.update(byteVec, 0, byteVec.size());
    vector<byte> out2;
    hash.hashFinal(out2, 0);

    //concatenate the two digests
    out.insert(out.end(), out2.begin(), out2.end());

    vector<vector<byte>> sendBufsBytes(N, out);
    recBufsBytes.resize(N);

    //now ssend the hash values to all servers
    //resize vectors
    for(int i=0; i < N; i++)
    {
        recBufsBytes[i].resize(sendBufsBytes[0].size());
    }

    //call the round function to send the shares to all the users and get the other parties share
    roundFunctionSync(sendBufsBytes, recBufsBytes,20);


}
template <class FieldType>
int ProtocolParty<FieldType>::generateClearMatricesForTesting(vector<FieldType> &accMsgsMat,
                                    vector<FieldType> &accMsgsSquareMat,
                                    vector<FieldType> &accCountersMat,
                                    vector<int> &accIntCountersMat){

    vector<vector<byte>> allHashes;

    auto t1 = high_resolution_clock::now();
    commitOnMatrices(accMsgsMat, accCountersMat,allHashes);
    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds commit on matrices: " << duration << endl;
    }
    //compute just the open mats for debuging purpuses

    vector<FieldType> accMsgsMatOpened(accMsgsMat.size());
    vector<FieldType> accMsgsSquareMatOpened(accMsgsMat.size());

    vector<FieldType> accCountersMatOpened(accCountersMat.size());


    vector<vector<byte>> accMsgsMatMatAll(N);
    vector<vector<byte>> accMsgsSquareMatAll(N);
    vector<vector<byte>> accCountersMatAll(N);


    openShareSetRecBuf(accMsgsMat.size(), accMsgsMat, accMsgsMatOpened, 2*T,accMsgsMatMatAll);
    openShareSetRecBuf(accMsgsSquareMat.size(), accMsgsSquareMat, accMsgsSquareMatOpened, 2*T,accMsgsSquareMatAll);


    openShareSetRecBuf(accCountersMat.size(), accCountersMat, accCountersMatOpened, 2*T, accCountersMatAll);

    for(int i=0; i<accCountersMat.size(); i++){

        accIntCountersMat[i] = accCountersMatOpened[i].elem;
    }

    accMsgsMat = move(accMsgsMatOpened);
    accMsgsSquareMat = move(accMsgsSquareMatOpened);

    //check that the hashes are in fact correct, that is, the servers committed on the right shares


//    for(int i=0; i<accMsgsMatOpened.size(); i++) {
//        cout << "value " << i << " is " << accMsgsMatOpened[i] << endl;
//    }
//    for(int i=0; i<accCountersMat.size(); i++) {
//        cout << "counter num " << i << " is " << accCountersMat[i] << endl;
//    }

    t1 = high_resolution_clock::now();
    OpenSSLSHA256 hash;
    vector<byte> out;
    vector<byte> out2;
    for(int i=0; i<numServers; i++){

        //hash the messsages and the counters shares
        hash.update(accMsgsMatMatAll[i], 0, accMsgsMatMatAll[i].size());
        out.resize(0);

        hash.hashFinal(out, 0);

        hash.update(accCountersMatAll[i], 0, accCountersMatAll[i].size());

        out2.resize(0);
        hash.hashFinal(out2, 0);

        //concatenate the two digests
        out.insert(out.end(), out2.begin(), out2.end());

        if(out!=allHashes[i]) {
            return i;
        }
    }

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in milliseconds second hash: " << duration << endl;
    }

    return -1;

}


template <class FieldType>
int ProtocolParty<FieldType>::generateClearMatrices(vector<FieldType> &accMats, vector<FieldType> &accFieldCountersMat,
                                                    vector<int> &accIntCountersMat){

    vector<vector<byte>> allHashes;

    commitOnMatrices(accMats, accFieldCountersMat,allHashes);

    //compute just the open mats for debuging purpuses

    vector<FieldType> accMatOpened(accMats.size());
    vector<FieldType> accFieldCountersMatOpened(accFieldCountersMat.size());


    vector<vector<byte>> accMatAll(N);
    vector<vector<byte>> accFieldCountersMatAll(N);
    openShareSetRecBuf(accMats.size(), accMats, accMatOpened, 2*T,accMatAll);

    openShareSetRecBuf(accFieldCountersMat.size(), accFieldCountersMat, accFieldCountersMatOpened, 2*T, accFieldCountersMatAll);

    for(int i=0; i<accFieldCountersMat.size(); i++){

        accIntCountersMat[i] = accFieldCountersMatOpened[i].elem;
    }

    accMats = move(accMatOpened);

    //check that the hashes are in fact correct, that is, the servers committed on the right shares


    for(int i=0; i<accMatOpened.size(); i++) {
        cout << "value " << i << " is " << accMatOpened[i] << endl;
    }
    for(int i=0; i<accFieldCountersMatOpened.size(); i++) {
        cout << "counter num " << i << " is " << accFieldCountersMatOpened[i] << endl;
    }

    OpenSSLSHA256 hash;
     vector<byte> out;
    vector<byte> out2;
    for(int i=0; i<numServers; i++){

           //hash the messsages and the counters shares
        hash.update(accMatAll[i], 0, accMatAll[i].size());
        out.resize(0);

        hash.hashFinal(out, 0);

        hash.update(accFieldCountersMatAll[i], 0, accFieldCountersMatAll[i].size());

        out2.resize(0);
        hash.hashFinal(out2, 0);

        //concatenate the two digests
        out.insert(out.end(), out2.begin(), out2.end());

        if(out!=allHashes[i]) {
            return i;
        }
    }

    return -1;

}


template<class FieldType>
void ProtocolParty<FieldType>::printOutputMessagesForTesting(vector<FieldType> &accMsgsMat,
                                                             vector<FieldType> &accMsgsMat2,
                                                             vector<int> &accIntCountersMat, int numMsgs){

    int counter = 0;

    for(int i=0; i<accIntCountersMat.size(); i++){

        if(i%10000==0) {
            cout << "accIntCountersMat[" << i << "]" << accIntCountersMat[i] << endl;
            if (accIntCountersMat[i] == 1) {
                for (int l1 = 0; l1 < l; l1++) {
                    cout << "\033[1;31mmessage #" << counter << " is " << accMsgsMat[l * i + l1] << "\033[0m" << endl;
                }
                counter++;
            } else if (accIntCountersMat[i] == 2) {

                for (int l1 = 0; l1 < l; l1++) {
                    cout << "\033[1;31mmessage #" << counter << " is " << accMsgsMat[l * i + l1] << "\033[0m" << endl;
                }
                counter++;

                for (int l1 = 0; l1 < l; l1++) {
                    cout << "\033[1;31mmessage #" << counter << " is " << accMsgsMat2[l * i + l1] << "\033[0m" << endl;
                }
                counter++;

            } else {
                //no messages to extract
            }
        }

    }

}

template<class FieldType>
void ProtocolParty<FieldType>::extractMessagesForTesting(vector<FieldType> &accMsgsMat,
                                               vector<FieldType> &accMsgsSquareMat,
                                               vector<int> &accIntCountersMat, int numMsgs){


    int totalNumber = 0;
    for (int i=0; i<numMsgs; i++){
        totalNumber += accIntCountersMat[i];
        for (int j=0; j<l;j++) {
            calcPairMessages(accMsgsMat[i * l + j], accMsgsSquareMat[i * l + j], accIntCountersMat[i], i, j);
        }
    }

    if (totalNumber > numClients){
        cout<<"CHEATING total counter number!!!"<<endl;
    }

    cout<<"sqrtR" <<sqrtR<<endl;
}


template<class FieldType>
void ProtocolParty<FieldType>::calcPairMessages(FieldType & a, FieldType & b, int counter, int i, int j){

    //If there is no element in this index, check that both values are zero.
    if (counter == 0){
        if (a != *(field->GetZero()) || b != *(field->GetZero())){
            cout<<"CHEATING counter == 0!!!"<<endl;
            cout<<"a = "<<a<<endl;
            cout<<"b = "<<b<<endl;
            cout<<"i = "<<i<<endl;
            cout<<"j = "<<j<<endl;
        }
        //If there is one element in this index, check that x = x^2.
    } else if (counter == 1){
        FieldType temp = a*a;
        if (b == temp){
            b = *(field->GetZero());
        } else {
            cout<<"CHEATING counter == 1!!!"<<endl;
            cout<<"i = "<<i<<endl;
            cout<<"j = "<<j<<endl;
        }
        //If there are two elements in this index, calculate them
    } else if (counter == 2){
        FieldType eight = field->GetElement(8);
        FieldType four = field->GetElement(4);
        FieldType two = field->GetElement(2);

        FieldType insideSqrt = eight*b - a*a*four; //8b-4a^2

        FieldType root = insideSqrt.sqrt();

        //calculate the final messages
        FieldType val = two*a;
        //put the first message in b
        a = (val + root) / four;

        //put the second message in a
        b = (val - root) / four;

    } else {
        a = *(field->GetZero());
        b = *(field->GetZero());
    }
}


template <class FieldType>
void ProtocolParty<FieldType>::offlineDNForMultiplication(int numOfTriples){

    generateRandom2TAndTShares(numOfTriples,randomTAnd2TShares);

}

template <class FieldType>
void ProtocolParty<FieldType>::inputPhase() {
    splitShiftFlat(msgsVectorsFlat, squaresVectorsFlat, countersVectorsFlat, unitVectorsFlat);
//    copyBackToVectors();
}

template <class FieldType>
int ProtocolParty<FieldType>::verificationPhase() {

    //first check that the inputs are consistent
    //checkInputConsistency(msgsVectors, unitVectors);



//    auto flag =  validMsgsTest(msgsVectors, unitVectors);
    auto flag =  validMsgsTestFlat((vector<FieldType>&)msgsVectorsFlat, (vector<FieldType>&)squaresVectorsFlat, (vector<FieldType>&)countersVectorsFlat, (vector<FieldType>&)unitVectorsFlat);

    cout<<"flag is : "<<flag<<endl;

    return flag;

  }


  template <class FieldType>
  vector<byte> ProtocolParty<FieldType>::generateCommonKey(){

      int fieldByteSize = field->getElementSizeInBytes();

      //calc the number of elements needed for 128 bit AES key
      int numOfRandomShares = 16/field->getElementSizeInBytes() + 1;
      vector<FieldType> randomSharesArray(numOfRandomShares);
      vector<FieldType> aesArray(numOfRandomShares);
      vector<byte> aesKey(numOfRandomShares*fieldByteSize);


      //generate enough random shares for the AES key

      getRandomShares(numOfRandomShares, randomSharesArray);

      openShare(numOfRandomShares, randomSharesArray, aesArray, T);


      //turn the aes array into bytes to get the common aes key.
      for(int i=0; i<numOfRandomShares;i++){

          for(int j=0; j<numOfRandomShares;j++) {
              field->elementToBytes(aesKey.data() + (j * fieldByteSize), aesArray[j]);
          }
      }

      //reduce the size of the key to 16 bytes
      aesKey.resize(16);

      return aesKey;

  }

template <class FieldType>
void ProtocolParty<FieldType>::openShare(int numOfRandomShares, vector<FieldType> &Shares, vector<FieldType> &secrets, int d){

    vector<vector<byte>> recBufsBytes(N);

    openShareSetRecBuf(numOfRandomShares, Shares, secrets, d, recBufsBytes);

}
  template <class FieldType>
  void ProtocolParty<FieldType>::openShareSetRecBuf(int numOfRandomShares, vector<FieldType> &Shares, vector<FieldType> &secrets,
          int d, vector<vector<byte>> &recBufsBytes){


      vector<vector<byte>> sendBufsBytes(N);

      vector<FieldType> x1(N);
      int fieldByteSize = field->getElementSizeInBytes();

      //calc the number of elements needed for 128 bit AES key

      //resize vectors
      for(int i=0; i < N; i++)
      {
          sendBufsBytes[i].resize(numOfRandomShares*fieldByteSize);
          recBufsBytes[i].resize(numOfRandomShares*fieldByteSize);
      }

      //set the first sending data buffer
      for(int j=0; j<numOfRandomShares;j++) {
          field->elementToBytes(sendBufsBytes[0].data() + (j * fieldByteSize), Shares[j]);
      }

      //copy the same data for all parties
      for(int i=1; i<N; i++){

          sendBufsBytes[i] = sendBufsBytes[0];
      }

      //call the round function to send the shares to all the users and get the other parties share
      roundFunctionSync(sendBufsBytes, recBufsBytes,12);

      //reconstruct each set of shares to get the secret

      for(int k=0; k<numOfRandomShares; k++){

          //get the set of shares for each element
          for(int i=0; i < N; i++) {

              x1[i] = field->bytesToElement(recBufsBytes[i].data() + (k*fieldByteSize));
          }


          secrets[k] = reconstructShare(x1, d);

      }

  }


template <class FieldType>
void ProtocolParty<FieldType>::generatePseudoRandomElements(vector<byte> & aesKey, vector<FieldType> &randomElementsToFill, int numOfRandomElements){


    int fieldSize = field->getElementSizeInBytes();
    int fieldSizeBits = field->getElementSizeInBits();
    bool isLongRandoms;
    int size;
    if(fieldSize>4){
      isLongRandoms = true;
      size = 8;
    }
    else{

      isLongRandoms = false;
      size = 4;
    }

    if (flag_print) {
        cout << "size is" << size << "for party : " << m_partyId;
    }


    PrgFromOpenSSLAES prg((numOfRandomElements*size/16) + 1);
    SecretKey sk(aesKey, "aes");
    prg.setKey(sk);


    for(int i=0; i<numOfRandomElements; i++){

      if(isLongRandoms)
          randomElementsToFill[i] = field->GetElement(((unsigned long)prg.getRandom64())>>(64 - fieldSizeBits));
      else
          randomElementsToFill[i] = field->GetElement(prg.getRandom32());
    }

}

/**
 * the function Walk through the circuit and reconstruct output gates.
 * @param circuit
 * @param gateShareArr
 * @param alpha
 */
template <class FieldType>
void ProtocolParty<FieldType>::outputPhase()
{

    vector<FieldType> accMsgsMat(sqrtR*sqrtU*l);
    vector<FieldType> accMsgsSquareMat(sqrtR*sqrtU*l);
    vector<FieldType> accCountersMat(sqrtR*sqrtU);
    vector<int> accIntCountersMat(sqrtR*sqrtU);

    auto t1 = high_resolution_clock::now();
    timer->startSubTask("generateSharedMatricesOptimized", iteration);
    generateSharedMatricesOptimizedFlat((vector<FieldType>&)msgsVectorsFlat,
                                        (vector<FieldType>&)squaresVectorsFlat,
                                        (vector<FieldType>&)countersVectorsFlat,
                                        (vector<FieldType>&)unitVectorsFlat,
                                        (vector<FieldType>&)accMsgsMat,
                                        (vector<FieldType>&)accMsgsSquareMat,
                                        (vector<FieldType>&)accCountersMat);
    timer->endSubTask("generateSharedMatricesOptimized", iteration);
    auto t2 = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings)
        cout << "time in miliseconds generateSharedMatricesOptimized: " << duration << endl;

    t1 = high_resolution_clock::now();
    timer->startSubTask("extractMessagesForTesting", iteration);

    int flag =  generateClearMatricesForTesting((vector<FieldType>&)accMsgsMat,
                                                (vector<FieldType>&)accMsgsSquareMat,
                                                (vector<FieldType>&)accCountersMat,
                                                accIntCountersMat);
    timer->endSubTask("extractMessagesForTesting", iteration);

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in miliseconds generateClearMatricesForTesting: " << duration << endl;
    }


    cout<<"flag for clear is "<<flag<<endl;

    if(flag==-1){

        cout<<"all hashes are correct"<<endl;
    }
    else
    {
        cout<<"basssssssssssssssssa you " <<flag <<endl;

    }

    t1 = high_resolution_clock::now();
    extractMessagesForTesting((vector<FieldType>&)accMsgsMat,
                              (vector<FieldType>&)accMsgsSquareMat,
                              accIntCountersMat,
                              accIntCountersMat.size());

    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in miliseconds extractMessagesForTesting: " << duration << endl;
    }


    t1 = high_resolution_clock::now();
    timer->startSubTask("printOutputMessagesForTesting", iteration);
    printOutputMessagesForTesting(accMsgsMat, accMsgsSquareMat, accIntCountersMat,numClients);
    timer->endSubTask("printOutputMessagesForTesting", iteration);
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    if(flag_print_timings) {
        cout << "time in miliseconds printOutputMessagesForTesting: " << duration << endl;
    }

    cout<<"passed with distinction"<<endl;
}

template <class FieldType>
void ProtocolParty<FieldType>::roundFunctionSync(vector<vector<byte>> &sendBufs, vector<vector<byte>> &recBufs, int round) {

    int numThreads = parties.size();
    int numPartiesForEachThread;

    if (parties.size() <= numThreads){
        numThreads = parties.size();
        numPartiesForEachThread = 1;
    } else{
        numPartiesForEachThread = (parties.size() + numThreads - 1)/ numThreads;
    }


    recBufs[m_partyId] = move(sendBufs[m_partyId]);
    //recieve the data using threads
    vector<thread> threads(numThreads);
    for (int t=0; t<numThreads; t++) {
        if ((t + 1) * numPartiesForEachThread <= parties.size())
            threads[t] = thread(&ProtocolParty::exchangeData, this, ref(sendBufs), ref(recBufs),
                                t * numPartiesForEachThread, (t + 1) * numPartiesForEachThread);
        else
            threads[t] = thread(&ProtocolParty::exchangeData, this, ref(sendBufs), ref(recBufs),
                    t * numPartiesForEachThread, parties.size());

    }

    for (int t=0; t<numThreads; t++)
        threads[t].join();
}


template <class FieldType>
void ProtocolParty<FieldType>::exchangeData(vector<vector<byte>> &sendBufs, vector<vector<byte>> &recBufs,
        int first, int last){

//    for (int i=first; i < last; i++) {
//
//        if ((m_partyId) < parties[i]->getID()) {
//
//
//            //send shares to my input bits
//            parties[i]->getChannel()->write(sendBufs[parties[i]->getID()].data(), sendBufs[parties[i]->getID()].size());
//            //cout<<"write the data:: my Id = " << m_partyId - 1<< "other ID = "<< parties[i]->getID() <<endl;
//
//
//            //receive shares from the other party and set them in the shares array
//            parties[i]->getChannel()->read(recBufs[parties[i]->getID()].data(), recBufs[parties[i]->getID()].size());
//            //cout<<"read the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID()<<endl;
//
//        } else{
//
//
//            //receive shares from the other party and set them in the shares array
//            parties[i]->getChannel()->read(recBufs[parties[i]->getID()].data(), recBufs[parties[i]->getID()].size());
//            //cout<<"read the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID()<<endl;
//
//
//
//            //send shares to my input bits
//            parties[i]->getChannel()->write(sendBufs[parties[i]->getID()].data(), sendBufs[parties[i]->getID()].size());
//            //cout<<"write the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID() <<endl;
//
//
//        }
//
//    }


}


template <class FieldType>
void ProtocolParty<FieldType>::roundFunctionSyncElements(vector<vector<FieldType>> &sendBufs,
        vector<vector<FieldType>> &recBufs, int round) {

    int numThreads = parties.size();
    int numPartiesForEachThread;

    if (parties.size() <= numThreads){
        numThreads = parties.size();
        numPartiesForEachThread = 1;
    } else{
        numPartiesForEachThread = (parties.size() + numThreads - 1)/ numThreads;
    }


    recBufs[m_partyId] = move(sendBufs[m_partyId]);
    //recieve the data using threads
    vector<thread> threads(numThreads);
    for (int t=0; t<numThreads; t++) {
        if ((t + 1) * numPartiesForEachThread <= parties.size()) {
            threads[t] = thread(&ProtocolParty::exchangeDataElements, this, ref(sendBufs), ref(recBufs),
                                t * numPartiesForEachThread, (t + 1) * numPartiesForEachThread);
        } else {
            threads[t] = thread(&ProtocolParty::exchangeDataElements, this, ref(sendBufs), ref(recBufs),
                    t * numPartiesForEachThread, parties.size());
        }
    }
    for (int t=0; t<numThreads; t++){
        threads[t].join();
    }

}


template <class FieldType>
void ProtocolParty<FieldType>::exchangeDataElements(vector<vector<FieldType>> &sendBufs,
        vector<vector<FieldType>> &recBufs, int first, int last) {

    this->roundFunctionSameMsg((byte*)recBufs.data(), (byte*)sendBufs.data(), sizeof(recBufs[0]));

//    for (int i = first; i < last; i++) {
//
//        if ((m_partyId) < parties[i]->getID()) {
//
//
//            if (sendBufs[parties[i]->getID()].size() > 0) {
//                //send shares to my input bits
//                parties[i]->getChannel()->write((byte *) sendBufs[parties[i]->getID()].data(),
//                                                sendBufs[parties[i]->getID()].size() * field->getElementSizeInBytes());
//                //cout<<"write the data:: my Id = " << m_partyId - 1<< "other ID = "<< parties[i]->getID() <<endl;
//            }
//
//            if (recBufs[parties[i]->getID()].size() > 0) {
//                //receive shares from the other party and set them in the shares array
//                parties[i]->getChannel()->read((byte *) recBufs[parties[i]->getID()].data(),
//                                               recBufs[parties[i]->getID()].size() * field->getElementSizeInBytes());
//                //cout<<"read the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID()<<endl;
//            }
//
//        } else {
//
//            if (recBufs[parties[i]->getID()].size() > 0) {
//                //receive shares from the other party and set them in the shares array
//                parties[i]->getChannel()->read((byte *) recBufs[parties[i]->getID()].data(),
//                                               recBufs[parties[i]->getID()].size() * field->getElementSizeInBytes());
//                //cout<<"read the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID()<<endl;
//            }
//
//            if (sendBufs[parties[i]->getID()].size() > 0) {
//
//                //send shares to my input bits
//                parties[i]->getChannel()->write((byte *) sendBufs[parties[i]->getID()].data(),
//                                                sendBufs[parties[i]->getID()].size() * field->getElementSizeInBytes());
//                //cout<<"write the data:: my Id = " << m_partyId-1<< "other ID = "<< parties[i]->getID() <<endl;
//            }
//
//        }
//
//    }
}


template <class FieldType>
ProtocolParty<FieldType>::~ProtocolParty()
{
    delete field;
    delete timer;
}


#endif /* PROTOCOLPARTY_H_ */
