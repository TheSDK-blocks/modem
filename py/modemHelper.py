"""
Created on Fri Jul 28 13:35:50 2017

@author: theluke
"""

import numpy as np
from scipy import signal
import scipy.io as sio

#Functions for (de)modulation

#Shape guidelines
#Users or antennas is first index, time is second index

#GolaySequence1 GolaySequence2 GolayGuardInterval FDEHeader 

def muSingleCarrierModulation(modulationParameterDict, golayParameterDict, pulseParameterDict, **keywordParameters):
    #Load modulation parameters
    #K: Number of users
    K = modulationParameterDict['K']
    oversamplingRatio = modulationParameterDict['OversamplingRatio']
    #A payload consists of only the transmit data, so no pilots
    oneUserPayloadLength = modulationParameterDict['OneUserPayloadLength']
    qamOrder = modulationParameterDict['QAMOrder']
    fdeFlag = modulationParameterDict['FDEFlag']
    golayFlag = modulationParameterDict['GolayFlag']
    pulseShape = generatePulseShape(pulseParameterDict)
    
    if 'GrayFlag' in modulationParameterDict:
        grayFlag = modulationParameterDict['GrayFlag']
    else:
        grayFlag = False
    
    #Load golay parameters
    #Golay pilot consists of 2 sequences right after one another
    #   and then a blank guard interval
    #Golay header consists of a blank interval of duration golayPilotLength*K
    #   One region of length golayPilotLength is populated with the Golay pilot
    #   The location of this region is based on the user index
    #   |           ||USER1GOLAY||                          |
    #   |                        ||USER2GOLAY||             |
    if golayFlag:
        golaySequenceLength = golayParameterDict['SequenceLength']
        golayGuardIntervalLength = golayParameterDict['GuardIntervalLength']
        golaySequenceA, golaySequenceB = generateGolaySequences(int(np.log2(golaySequenceLength)))
        golayPilotLength = 2*golaySequenceLength+golayGuardIntervalLength
        golayHeaderLength = K*golayPilotLength
    
    #Load optional FDE parameters
    #FDE is for zero-forcing the inter-user interference matrix at each subcarrier
    #Just like Golay, pilots are sent from users in a TDD fashion
    #This is a DFT-based equalization method, so guard intervals are used for each pilot block
    #   If FDE is enabled, then the payload is separated into blocks of the same length as the pilots
    #       and a cyclic prefix is applied to each of these
    if fdeFlag:
        fdeBlockLength = modulationParameterDict['FDEBlockLength']
        fdeCyclicPrefixLength = modulationParameterDict['FDECyclicPrefixLength']
        numFdePilots = modulationParameterDict['NumFDEPilots']
        
        #TX payload has a length equal to an integer number of FDE blocks
        oneUserPayloadNumFdeBlocks = int(np.ceil(oneUserPayloadLength/fdeBlockLength))
        oneUserPayloadLength = oneUserPayloadNumFdeBlocks*fdeBlockLength
        
        #Each data packet which will be sent through the air includes both the block and cyclic prefix
        fdeBlockAndCPLength = fdeBlockLength + fdeCyclicPrefixLength
        
    #Compute the length of each data packet
    oneUserPacketLength = 0
    
    if fdeFlag:
        oneUserPacketLength += fdeBlockAndCPLength*(oneUserPayloadNumFdeBlocks+K*numFdePilots)
    else:
        oneUserPacketLength += oneUserPayloadLength
    
    if golayFlag:
        oneUserPacketLength += golayHeaderLength
        
    #Modulate each user's data sequence
    allUsersPayloadWords = np.zeros((K, oneUserPayloadLength), dtype=int)
    allUsersPayloadSymbols = np.zeros((K, oneUserPayloadLength), dtype=complex)
    allUsersFdePilotSubcarrierWords = np.zeros((), dtype=int)
    allUsersPacketSymbols = np.zeros((K, oneUserPacketLength), dtype=complex)
    allUsersPacketSamples = np.zeros((K, oversamplingRatio*oneUserPacketLength), dtype=complex)
#    allUsersFdePilotSymbols = []

    if fdeFlag:
        #This contains the pilot sequences without cyclic prefixes
        allUsersFdePilotSubcarrierWords = np.zeros((K, numFdePilots, fdeBlockLength), dtype=int)
    if golayFlag:
        allUsersGolayPilots = np.zeros((K, golayPilotLength), dtype=complex)
    
    for userIndex in range(0, K):
        #Generate user payloads as QAM symbols
        if 'UserPayloads' in keywordParameters:
#            print("Using specified user payloads")
            oneUserPayloadBits = keywordParameters['UserPayloads'][userIndex,:]
        else:
            oneUserPayloadBits = np.random.randint(0, 2, oneUserPayloadLength*int(np.log2(qamOrder)))
        oneUserPayloadWords, oneUserPayloadSymbols = qamModulateBitStream(oneUserPayloadBits, qamOrder, grayFlag)
        
        
        if golayFlag:
            #Create Golay pilot by taking both sequences and the guard interval and combining
            oneUserGolayPilot = np.append(golaySequenceA, golaySequenceB)
            oneUserGolayPilot = np.append(oneUserGolayPilot, np.zeros((1, golayGuardIntervalLength), dtype=oneUserGolayPilot.dtype))
            allUsersGolayPilots[userIndex, :] = oneUserGolayPilot
            
            #Golay pilots are TDD, so the Golay header duration is the 
            #   time it takes for every user to send their Golay sequences
            #Initialize the Golay header as a (headerLength x K) matrix and
            #   only populate the current user's interval
            oneUserGolayHeader = np.zeros((K, golayPilotLength))
            oneUserGolayHeader[userIndex, :]  = oneUserGolayPilot
            oneUserGolayHeader = oneUserGolayHeader.flatten()
        
        #Add FDE cyclic prefix to the TX payload
        if fdeFlag:
            #################################################
            ######      PAYLOAD GENERATION          #########
            #################################################
            #Divide the payload into blocks of length fdeBlockLength
            #Then, prepend the cyclic prefix to the beginning of each of these blocks
            #   The cyclic prefix consists of the last fdeCyclicPrefixLength symbols
            oneUserFdePayloadBlocks = oneUserPayloadSymbols.reshape((-1, fdeBlockLength))
            cyclicPrefix = oneUserFdePayloadBlocks[:, -fdeCyclicPrefixLength:]
            oneUserFdePayloadBlocks = np.append(cyclicPrefix, oneUserFdePayloadBlocks, axis=1)
            oneUserPayloadWithCPSymbols = oneUserFdePayloadBlocks.flatten()
            
            ##################################################
            ######      PILOT GENERATION            ##########
            ##################################################
            #Generate pilots consisting of BPSK symbols for each subcarrier
            #   as well as a cyclic prefix which is prepended to the beginning of the pilot region
            if 'FDEPilots' in keywordParameters:
#                print("Using specified FDE pilots")
                oneUserFdePilotSubcarrierWords = keywordParameters['FDEPilots'][userIndex,:]
            else:
                oneUserFdePilotSubcarrierWords = np.random.randint(0, 2, (numFdePilots, fdeBlockLength))
            oneUserFdePilotSubcarrierSymbols = qamMod(oneUserFdePilotSubcarrierWords.flatten(), 2)
            oneUserFdePilotSubcarrierSymbols = oneUserFdePilotSubcarrierSymbols.reshape((-1, fdeBlockLength))
            
            allUsersFdePilotSubcarrierWords[userIndex, :, :] = oneUserFdePilotSubcarrierWords
            
            oneUserFdePilotSymbols = np.sqrt(fdeBlockLength)*np.fft.ifft(oneUserFdePilotSubcarrierSymbols, axis=1)
#            allUsersFdePilotSymbols[userIndex, :] = oneUserFdePilotSymbols.flatten()
            
            cyclicPrefix = oneUserFdePilotSymbols[:, -fdeCyclicPrefixLength:]
            
            oneUserFdePilotSymbols = np.append(cyclicPrefix, oneUserFdePilotSymbols, axis=1)
            oneUserFdePilotSymbols = oneUserFdePilotSymbols.flatten()
#            allUsersFdePilotSymbols[userIndex, :] = oneUserFdePilotSymbols
            
            #Pilot region should be all zeros, except for when userIndex is transmitting its pilot
            oneUserFdeHeaderSymbols = np.zeros((K, fdeBlockAndCPLength*numFdePilots), dtype=complex)
            oneUserFdeHeaderSymbols[userIndex, :] = oneUserFdePilotSymbols
            oneUserFdeHeaderSymbols = oneUserFdeHeaderSymbols.flatten()
    
        oneUserPacketSymbols = oneUserPayloadWithCPSymbols
        if fdeFlag:
            oneUserPacketSymbols = np.append(oneUserFdeHeaderSymbols, oneUserPacketSymbols)
        if golayFlag:
            oneUserPacketSymbols = np.append(oneUserGolayHeader, oneUserPacketSymbols)
            
        oneUserPacketSamples = oversampleWaveform(oneUserPacketSymbols, oversamplingRatio, pulseShape)
        
        allUsersPacketSymbols[userIndex, :] = oneUserPacketSymbols
        allUsersPacketSamples[userIndex, :] = oneUserPacketSamples
        allUsersPayloadWords[userIndex, :] = oneUserPayloadWords
        allUsersPayloadSymbols[userIndex, :] = oneUserPayloadSymbols
        
    return allUsersPayloadWords, allUsersFdePilotSubcarrierWords, allUsersPacketSamples

def golayChannelAndTimingEstimator(oversampledNormalizedCorrelation, corrPhase0, corrPhase1, numUsers, golayParameterDict):    
    #For each user, find the peaks of the correlation sequence, which is currently 2x oversampled
    allUsersCorrelationPeakLocation, allUsersSampleOffset = findCorrelationPeaksPerUser(oversampledNormalizedCorrelation, golayParameterDict, numUsers)
    
    numInterpSamples = 41
    allUsersDelayEstimate = np.zeros((numUsers,))
    allUsersSequenceMaxInterp = np.zeros((numUsers, numInterpSamples))
    allUsersSampleOffsetEstimate = np.zeros((numUsers,), dtype=int)
    allUsersChannelGain = np.zeros((numUsers,), dtype=complex)
    allUsersDetectionFlag = np.zeros((numUsers,), dtype=bool)
    
    #For each user, find the fractional symbol delay
    #This is done by creating an interpolation polynomial +- 1 data symbol (not oversampled) around the correlation peak,
    #   then finding the maximum of this polynomial
    for userIndex in range(0, numUsers):
        maximumDetectedIndex = allUsersCorrelationPeakLocation[userIndex]
        
        #Keeps track of which oversampled phase the peak is found on, since we want
        #   all antenna signals to be referenced to the same oversampled clock phase
        phase1Flag = False
        
        #Create a sequence of 5 oversampled points around the correlation peak, 
        #   which will then be used for interpolation
        if (maximumDetectedIndex % 2) == 0:
            dataStartIndex = int(maximumDetectedIndex/2)
            correlationSequence = np.array([corrPhase0[dataStartIndex-1], corrPhase1[dataStartIndex-1], 
                                            corrPhase0[dataStartIndex], corrPhase1[dataStartIndex], 
                                            corrPhase0[dataStartIndex+1]])
        else:
            dataStartIndex = int((maximumDetectedIndex-1)/2)
            correlationSequence = np.array([corrPhase1[dataStartIndex-1], corrPhase0[dataStartIndex], 
                                            corrPhase1[dataStartIndex], corrPhase0[dataStartIndex+1], 
                                            corrPhase1[dataStartIndex+1]])
            phase1Flag = True

        #MODIFICATION NEEDED: WHAT IS SAMPLE OFFSET ACTUALLY USED FOR?
        allUsersSampleOffsetEstimate[userIndex] = int(np.floor(allUsersSampleOffset[userIndex]/2))
        
        #Find the fractional sample maximum of the power of the correlation sequence
        sequenceMax = np.abs(correlationSequence)**2
        #Number of iterations used for the Newton's method
        numIterations = golayParameterDict['NumIters']
        oneUserDelayEstimate = refineDelayEstimate(numIterations, sequenceMax)

        #Shift correlation sequence by the estimated delay
        #This means that index 2 will be the true power peak of the correlation sequence
        #Take the complex value of the correlation sequence at this point, 
        #   which is the channel gain for a given user
        interpolatedCorrelationSequence = shiftTiming5(correlationSequence, oneUserDelayEstimate)
        allUsersChannelGain[userIndex] = interpolatedCorrelationSequence[2]
        allUsersDetectionFlag[userIndex] = True
        
        #Always want to shift the peak of correlation to land on the first phase
        #If the correlation peak is closer to the second phase than the first,
        #   then there are ways this could happen:
        #       Peak closer to second phase and comes x UI before the second phase
        #           Shift backward by 1+x
        #       Peak closer to second phase and comes x UI after the second phase
        #           Shift forward by 1-x
        if phase1Flag:
            if oneUserDelayEstimate < 0:
                oneUserDelayEstimate = (1+oneUserDelayEstimate)
            else:
                oneUserDelayEstimate = -(1-oneUserDelayEstimate)
        allUsersDelayEstimate[userIndex] = oneUserDelayEstimate
        
        #This is for debugging; it saves the interpolated polynomial for graphing
        lagrangePolynomialCoeffs = lagrangePolynomial(sequenceMax)
        sequenceMaxInterp = polynomialEvaluate(lagrangePolynomialCoeffs, np.linspace(-2, 2, numInterpSamples))
        allUsersSequenceMaxInterp[userIndex, :] = sequenceMaxInterp
        
    return allUsersDetectionFlag, allUsersDelayEstimate, allUsersSampleOffsetEstimate, allUsersChannelGain, allUsersSequenceMaxInterp

def findCorrelationPeaksPerUser(peakDetect2x, golayParameterDict, numUsers):
    golaySequenceLength = golayParameterDict['SequenceLength']
    golayGuardIntervalLength = golayParameterDict['GuardIntervalLength']
    golayPilotSequenceLength = 2*golaySequenceLength + golayGuardIntervalLength
    
    allUsersCorrelationPeakLocation = np.zeros((numUsers,))
    allUsersSampleOffset = np.zeros((numUsers,))
    for userIndex in range(0, numUsers):
        #Look only in the range of samples where the current user's Golay correlation is expected to be
        #This may not exactly align with the actual Golay correlation window (if there a delay on the user or ADC),
        #   but this is fine because the delay on a user will never be large enough to move the correct correlation peak outside
        #   of this window. If a user has a large delay, then they will be told (not yet implemented) to delay or advance their data
        oneUserCorrelationIndexWindow = range(userIndex*2*golayPilotSequenceLength, (userIndex+1)*2*golayPilotSequenceLength)
        oneUserCorrelation = peakDetect2x[oneUserCorrelationIndexWindow]
        
        #Find the peak in this window, which should be this user's correlation peak location
        oneUserCorrelationPeakLocation = np.argmax(oneUserCorrelation)
        
        allUsersSampleOffset[userIndex] = oneUserCorrelationPeakLocation - 2*(golaySequenceLength-1)
        allUsersCorrelationPeakLocation[userIndex] = oneUserCorrelationPeakLocation + userIndex*2*golayPilotSequenceLength
    
    return allUsersCorrelationPeakLocation, allUsersSampleOffset

#Currently a 2-phase Golay correlator, where 2x oversampled data is correlated with
#   known Golay sequences for even and odd phases
def polyphaseGolayCorrelator(golayParameterDict, inputSeq):
    golaySequenceLength = golayParameterDict['SequenceLength']
    
    #Golay sequences are deterministically created based only on their desired length
    golayA, golayB = generateGolaySequences(int(np.log2(golaySequenceLength)))
    
    #Extract the two Golay sequence regions in the input sequence and correlate them
    #   with the known Golay sequences
    #The sum of autocorrelation of golayA and the autocorrelation of golayB is an impulse
    #This makes it very good for determining a user's delay
    
    #Phase0 is the even phase of the oversampled data
    seqAPhase0 = inputSeq[0:-2*golaySequenceLength:2]
    seqBPhase0 = inputSeq[2*golaySequenceLength::2]
    #Ideally, this is an impulse
    corrPhase0 = 1/2/golaySequenceLength*(np.convolve(np.flip(golayA, axis=0),seqAPhase0)+np.convolve(np.flip(golayB, axis=0),seqBPhase0))
    
    #Phase1 is the odd phase of the oversampled data
    seqAPhase1 = inputSeq[1:-2*golaySequenceLength:2]
    seqBPhase1 = inputSeq[2*golaySequenceLength+1::2]
    #Ideally this is an impulse
    corrPhase1 = 1/2/golaySequenceLength*(np.convolve(np.flip(golayA, axis=0),seqAPhase1)+np.convolve(np.flip(golayB, axis=0),seqBPhase1))
    
    corrPowerPhase0 = np.abs(corrPhase0)**2
    corrPowerPhase1 = np.abs(corrPhase1)**2
    
    #Interleave the two peak detections
    oversampledCorrelationPower = np.transpose(np.array([corrPowerPhase0, corrPowerPhase1])).flatten()
    
    return oversampledCorrelationPower, corrPhase0, corrPhase1

#Find the fractional sample delay of a user based on interpolating sequence 
#   of 5 2x oversampled points with indices [-2, -1, 0, 1, 2]
#0 corresponds to the estimated peak
def refineDelayEstimate(numIterations, inputSeq):
    lagrangePolynomialCoeffs = lagrangePolynomial(inputSeq)
    derivLagrangePolynomialCoeffs = np.array([4, 3, 2, 1])*lagrangePolynomialCoeffs[0:4]
    
    #Find roots of the Lagrange polynomial's derivative, as this corresponds to the max
    delayEstimation = newtonMethodRootPow2(3, numIterations, 1e-3, 0, derivLagrangePolynomialCoeffs)
    
    return delayEstimation

#Create a Lagrange polynomial interpolated sequence using 5 samples
#a is the coefficient of the highest power of x
def lagrangePolynomial(inputSeq):
    #Assumes that t=0 is index 2
    
    a = 1*inputSeq[0]/24        +   1*inputSeq[1]/(-6)      +   1*inputSeq[2]/4         +   1*inputSeq[3]/(-6)          +   1*inputSeq[4]/24
    b = (-2)*inputSeq[0]/24     +   (-1)*inputSeq[1]/(-6)   +   0*inputSeq[2]/4         +   (1)*inputSeq[3]/(-6)        +   (2)*inputSeq[4]/24
    c = (-1)*inputSeq[0]/24     +   (-4)*inputSeq[1]/(-6)   +   (-5)*inputSeq[2]/4      +   (-4)*inputSeq[3]/(-6)       +   (-1)*inputSeq[4]/24
    d = (2)*inputSeq[0]/24      +   (4)*inputSeq[1]/(-6)    +   (0)*inputSeq[2]/4       +   (-4)*inputSeq[3]/(-6)       +   (-2)*inputSeq[4]/24
    e = (0)*inputSeq[0]/24      +   (0)*inputSeq[1]/(-6)    +   (4)*inputSeq[2]/4       +   (0)*inputSeq[3]/(-6)        +   (0)*inputSeq[4]/24
    
    return [a, b, c, d, e]

#Evaluate a polynomial given its coefficients and an input vector
#polynomialCoeffs[0] is assumed to be the coefficient of the highest power of x
def polynomialEvaluate(polynomialCoeffs, x):
    polyOrder = len(polynomialCoeffs)-1
    interpOutput = np.zeros((len(x),))
    for coeffIndex in range(0, polyOrder):
        interpOutput = interpOutput + x**(polyOrder-coeffIndex)*polynomialCoeffs[coeffIndex]
    
    return interpOutput

#Use 5th order lagrange interpolation to shift the input sequence by delay (positive or negative) td
def shiftTiming5(inputSeq, td):
    coeffs = np.array([(td+1)*td*(td-1)*(td-2)/24, (td+2)*td*(td-1)*(td-2)/-6, (td+2)*(1+td)*(td-1)*(td-2)/4, (td+2)*(1+td)*td*(td-2)/-6, (td+2)*(1+td)*(td)*(td-1)/24])
    seq_shift = np.convolve(np.flip(coeffs, axis=0), inputSeq)
    seq_true = seq_shift[3:-4]
    
    samp1 = polynomialInterp(td-1, inputSeq[0:4])
    samp_m1 = polynomialInterp(td+1, inputSeq[-4:])
    samp_m0 = polynomialInterp(td+2, inputSeq[-4:])
    
    outseq = np.append(samp1, seq_true)
    outseq = np.append(outseq, (samp_m1, samp_m0))
    return outseq

def polynomialInterp(td, seq):
    coeffs = np.array([td*(td-1)*(td-2)/-6, (1+td)*(td-1)*(td-2)/2, (1+td)*td*(td-2)/-2, (1+td)*(td)*(td-1)/6])
    
    outValue = np.dot(coeffs, seq)
    return outValue
    
#Use Newton's method to find the root of an input polynomial with coefficients functionCoefficients
#functionCoefficients[0] is the coefficient of the highest power of x
def newtonMethodRootPow2(functionOrder, numIterations, rootErrorThreshold, initialGuess, functionCoefficients):
    xI = initialGuess
    
    if numIterations > 0:
        for iterationIndex in range(0, numIterations):
            functionValue = np.dot(xI**np.arange(functionOrder, -1, -1), functionCoefficients)
            derivativeValue = np.dot(xI**np.arange(functionOrder-1, -1, -1), functionCoefficients[0:-1]*np.array([3, 2, 1]))
            
            #Find the closest power of 2 to the derivative
            #This allows functionValue/derivativeValueRounded to be a simple bit shift in hardware
            derivativeValueRounded = np.sign(derivativeValue)*2**np.round(np.log2(np.abs(derivativeValue)))

            xI = xI - functionValue/derivativeValueRounded
    else:
        correctionValue = rootErrorThreshold + 1
        iterationCount = 0
        while (correctionValue > rootErrorThreshold) or (iterationCount < 1000):
            functionValue = np.dot(xI**np.arange(functionOrder, -1, -1), functionCoefficients)
            derivativeValue = np.dot(xI**np.arange(functionOrder-1, -1, -1), functionCoefficients[0:-1]*np.array([3, 2, 1]))
            
            #Find the closest power of 2 to the derivative
            derivativeValueRounded = np.sign(derivativeValue)*2**np.round(np.log2(np.abs(derivativeValue)))
            
            correctionValue = functionValue/derivativeValueRounded
            xI = xI - correctionValue
            
            iterationCount += 1
            
    root = xI
    return root
            
def nextPow2(inputValue):
    nextValue = np.ceil(np.log2(inputValue))
    return nextValue

#Correlate an input sequence with the Golay sequences to find which sampling
#   phase is best for decimation
#Here, an oversampling ratio of 2 is assumed
def pickSamplingPhaseGolay(inputSeq, golayParameterDict, K):
    #Select only the region where there should be Golay pilots
    #Length of a single user's Golay pilot
    golayPilotLength = golayParameterDict['GolayPilotLength']
    #The data is still 2x oversampled, so multiply golayPilotLength by 2
    golayPilotsWindowIndex = range(0, K*2*golayPilotLength)
    golayPilotsWindow = inputSeq[golayPilotsWindowIndex]
    
    _, corrPhase0, corrPhase1 = polyphaseGolayCorrelator(golayParameterDict, golayPilotsWindow)
    
    #Use the phase which has the highest correlation peak for decimation
    peakCorrPhase0 = np.max(np.abs(corrPhase0)**2)
    peakCorrPhase1 = np.max(np.abs(corrPhase1)**2)
    if peakCorrPhase0 > peakCorrPhase1:
        oneUserCorrectlySampledSymbols = inputSeq[0::2]
    else:
        oneUserCorrectlySampledSymbols = inputSeq[1::2]
        
    return oneUserCorrectlySampledSymbols

def qamModulateBitStream(bitStream, qamOrder, grayFlag=False):
    #bitStream should be a numpy binary array
    if qamOrder > 256:
        print("Symbols with >8 bits currently not supported!")
        return 0
    #Generate QAM constellation of order qamOrder
    qamConstellation = qamMod(range(0, qamOrder), qamOrder)
    
    #Scale constellation to have an average power of 1
    normalizationScalar = modNorm(qamConstellation, 'avpow')
    
    bitsPerSymbol = int(np.log2(qamOrder))
    
    bitStreamSymbolGrouped = bitStream.reshape((-1, bitsPerSymbol))
    numSymbols = bitStreamSymbolGrouped.shape[0]
    bitStreamBytePad = np.zeros((numSymbols, 8-bitsPerSymbol), dtype='int32')
    wordStream = np.packbits(np.append(bitStreamBytePad, bitStreamSymbolGrouped, axis=1)).astype(int)
    
    qamSymbolStream = normalizationScalar*qamMod(wordStream, qamOrder, grayFlag)
    
    return wordStream, qamSymbolStream

def qamDemod(symbolStream, qamOrder, grayFlag=False):
    realMaxValue = int(np.sqrt(qamOrder)-1)
    
    realWordStream = np.clip(np.round((np.real(symbolStream)+realMaxValue)/2), 0, realMaxValue).astype(int)
    imagWordStream = np.clip(np.round((np.imag(symbolStream)+realMaxValue)/2), 0, realMaxValue).astype(int)
    
    if grayFlag:
        realWordStream = grayDemod(realWordStream)
        imagWordStream = grayDemod(imagWordStream)
    
    wordStream = imagWordStream + (realMaxValue+1)*realWordStream
    
    return wordStream

def qamMod(wordStream, qamOrder, grayFlag=False):
    if qamOrder == 2:
        qamDictionary = np.array([-1, 1])
    else:
        #wordStream between 0 and qamOrder-1
        realMaxValue = int(np.sqrt(qamOrder))-1
        qamDictionaryI = np.arange(-realMaxValue, realMaxValue+2, 2)
        
        qamDictionary = np.zeros((len(qamDictionaryI), len(qamDictionaryI)), dtype=complex)
        for iIndex in range(0, len(qamDictionaryI)):
            iValue = qamDictionaryI[iIndex]
            for qIndex in range(0, len(qamDictionaryI)):
                qValue = qamDictionaryI[qIndex]
                qamDictionary[iIndex,qIndex] = iValue + 1j*qValue            
        qamDictionary = qamDictionary.flatten()
        
        if grayFlag:
            numRealBits = int(np.log2(np.sqrt(qamOrder)))
            wordStreamI = np.bitwise_and(wordStream, 2**numRealBits-1)
            wordStreamQ = np.right_shift(wordStream, numRealBits)
            
            wordStreamI = grayMod(wordStreamI)
            wordStreamQ = grayMod(wordStreamQ)
            
            wordStream = wordStreamI + np.left_shift(wordStreamQ, numRealBits)
        
    symbolStream = qamDictionary[wordStream]
    return symbolStream

#Use recursive algorithm to create a Gray code lookup table
#Outputs an nparray
def grayCodeList(numBits):
    codeList = np.array([0,1])
    numIters = numBits-1
    nextPow2 = 2*1
    
    while numIters:
        codeList = np.append(codeList, nextPow2+np.flip(codeList,0))
        numIters = numIters-1
        nextPow2 = 2*nextPow2
    
    return codeList

def grayMod(wordStream):
    return np.bitwise_xor(wordStream, np.right_shift(wordStream, 1))

#This is valid for up to 8 bits, which would be insane
#   for any real/imag part of any QAM sequence
def grayDemod(wordStream):
    wordStream = np.bitwise_xor(wordStream, np.right_shift(wordStream, 4))
    wordStream = np.bitwise_xor(wordStream, np.right_shift(wordStream, 2))
    wordStream = np.bitwise_xor(wordStream, np.right_shift(wordStream, 1))
    
    return wordStream

def modNorm(symbolStream, powerMetric):
    if powerMetric == 'avpow':
        meanPower = np.mean(np.abs(symbolStream)**2)
        normalizingFactor = np.sqrt(1/meanPower)
    
    return normalizingFactor

def generateGolaySequences(N):
    a = np.array(1)
    b = np.array(1)
    
    for i in range(0,N):
        oldA = a
        oldB = b
        a = np.append(oldA, oldB)
        b = np.append(oldA, -oldB)
        
    golayA = a
    golayB = b
    
    return golayA, golayB

def generatePulseShape(pulseParameterDict):
    pulseType = pulseParameterDict['PulseType']
    oversamplingRatio = pulseParameterDict['OversamplingRatio']
    if pulseType == 'rrc':
        filtSpan = pulseParameterDict['FiltSpan']
        beta = pulseParameterDict['Beta']
        tRrcFilt = np.linspace(-filtSpan, filtSpan, num=2*oversamplingRatio*filtSpan+1, endpoint=True)
        tRrcFilt = np.where(tRrcFilt==0.0, 1.0e-10, tRrcFilt)
        tRrcFilt = np.where(tRrcFilt==1/4/beta, 1/4/beta+1.0e-10, tRrcFilt)
        tRrcFilt = np.where(tRrcFilt==-1/4/beta, 1/4/beta-1.0e-10, tRrcFilt)
        pulseShape = (np.sin(np.pi*tRrcFilt*(1-beta))+4*beta*tRrcFilt*np.cos(np.pi*tRrcFilt*(1+beta)))/(np.pi*tRrcFilt*(1-(4*beta*tRrcFilt)**2))/np.sqrt(oversamplingRatio)
    elif pulseType == 'rc':
        filtSpan = pulseParameterDict['FiltSpan']
        beta = pulseParameterDict['Beta']
        tRcFilt = np.linspace(-filtSpan, filtSpan, num=2*oversamplingRatio*filtSpan+1, endpoint=True)
        tRcFilt = np.where(tRcFilt==1.0, 1.0+1.0e-10, tRcFilt)
        tRcFilt = np.where(tRcFilt==-1.0, -1.0-1.0e-10, tRcFilt)
        pulseShape = np.sinc(tRcFilt)*np.cos(np.pi*beta*tRcFilt)/(1-4*beta**2*tRcFilt**2)
    
    return pulseShape


#DO I SERIOUSLY NEED TO WRITE THIS FUCKING FUNCTION???!?!?!
def oversampleWaveform(inputSamples, oversamplingRatio, pulseShape):
    outputWaveform = np.zeros((len(inputSamples)*oversamplingRatio), dtype=complex)
    outputWaveform[0:len(outputWaveform):oversamplingRatio] = inputSamples
    
    outputWaveform = np.convolve(outputWaveform, pulseShape)
    outputWaveform = outputWaveform[int((len(pulseShape)-1)/2):-int((len(pulseShape)-1)/2)]
    
    return outputWaveform