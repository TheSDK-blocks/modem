"""
Created on Fri Jul 28 13:35:50 2017

@author: theluke
"""

import numpy as np
from scipy import signal

#Functions for (de)modulation

#Shape guidelines
#Users or antennas is first index, time is second index

#GolaySequence1 GolaySequence2 GolayGuardInterval FDEHeader 

def muSingleCarrierModulation(modulationParameterDict, pulseShape, golayParameterDict, **keywordParameters):
    #Load modulation parameters
    #K: Number of users
    K = modulationParameterDict['K']
    oversamplingRatio = modulationParameterDict['OversamplingRatio']
    #A payload consists of only the transmit data, so no pilots
    oneUserPayloadLength = modulationParameterDict['OneUserPayloadLength']
    qamOrder = modulationParameterDict['QAMOrder']
    fdeFlag = modulationParameterDict['FDEFlag']
    golayFlag = modulationParameterDict['GolayFlag']
    
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
        allUsersFdePilotSubcarrierWords = np.zeros((K, fdeBlockLength*numFdePilots), dtype=int)
    if golayFlag:
        allUsersGolayPilots = np.zeros((K, golayPilotLength), dtype=complex)
    
    for userIndex in range(0, K):
        #Generate user payloads as QAM symbols
        if 'UserPayloads' in keywordParameters:
#            print("Using specified user payloads")
            oneUserPayloadBits = keywordParameters['UserPayloads'][userIndex,:]
        else:
            oneUserPayloadBits = np.random.randint(0, 2, oneUserPayloadLength*int(np.log2(qamOrder)))
        oneUserPayloadWords, oneUserPayloadSymbols = qamModulateBitStream(oneUserPayloadBits, qamOrder)
        
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
                oneUserFdePilotSubcarrierWords = np.random.randint(0, 2, numFdePilots*fdeBlockLength)
            oneUserFdePilotSubcarrierSymbols = qamMod(oneUserFdePilotSubcarrierWords, 2)
            oneUserFdePilotSubcarrierSymbols = oneUserFdePilotSubcarrierSymbols.reshape((-1, fdeBlockLength))
            
            allUsersFdePilotSubcarrierWords[userIndex, :] = oneUserFdePilotSubcarrierWords
            
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
        
    return allUsersPayloadWords, allUsersFdePilotSubcarrierWords, allUsersPacketSamples, allUsersPacketSymbols, allUsersPayloadSymbols

def muSingleCarrierDemodulation(modulationParameterDict, pulseShape, 
                                allUsersFdePilotSubcarrierWords, allUsersPacketSamples, 
                                golayParameterDict):
    
    #Load modulation parameters
    K = modulationParameterDict['K']
    oversamplingRatio = modulationParameterDict['OversamplingRatio']
    oneUserPayloadLength = modulationParameterDict['OneUserPayloadLength']
    qamOrder = modulationParameterDict['QAMOrder']
    fdeFlag = modulationParameterDict['FDEFlag']
    golayFlag = modulationParameterDict['GolayFlag']
    
    #Load golay parameters
    if golayFlag:
        golaySequenceLength = golayParameterDict['SequenceLength']
        golayGuardIntervalLength = golayParameterDict['GuardIntervalLength']
        golaySequenceA, golaySequenceB = generateGolaySequences(int(np.log2(golaySequenceLength)))
        golayPilotLength = 2*golaySequenceLength+golayGuardIntervalLength
        golayHeaderLength = K*golayPilotLength
    
    #Load FDE parameters
    if fdeFlag:
        fdeBlockLength = modulationParameterDict['FDEBlockLength']
        fdeCyclicPrefixLength = modulationParameterDict['FDECyclicPrefixLength']
        numFdePilots = modulationParameterDict['NumFDEPilots']
        
        #TX payload has a length equal to an integer number of FDE blocks
        oneUserPayloadNumFdeBlocks = int(np.ceil(oneUserPayloadLength/fdeBlockLength))
        oneUserPayloadLength = oneUserPayloadNumFdeBlocks*fdeBlockLength
        
        #Each data packet which is sent through the air includes both the block and cyclic prefix
        fdeBlockAndCPLength = fdeBlockLength + fdeCyclicPrefixLength
        
    #Compute the length of each data packet and the starting point of user payloads
    oneUserPacketLength = 0
    oneUserFdePilotStartIndex = 0
    oneUserPayloadStartIndex = 0
    
    if golayFlag:
        oneUserPacketLength += golayHeaderLength
        oneUserPayloadStartIndex += golayHeaderLength
        oneUserFdePilotStartIndex += golayHeaderLength
    
    if fdeFlag:
        oneUserPacketLength += fdeBlockAndCPLength*(oneUserPayloadNumFdeBlocks+K*numFdePilots)
        oneUserPayloadStartIndex += K*numFdePilots*fdeBlockAndCPLength
    else:
        oneUserPacketLength += oneUserPayloadLength
    
#    allUsersPacketSymbols = np.zeros((K, oneUserPacketLength+1), dtype=complex)
#    allUsersPayloadWords = np.zeros((K, oneUserPayloadLength), dtype=int)
#    allUsersSamplesFiltered = np.zeros((K, oneUserPacketLength*oversamplingRatio+2), dtype=complex)
#    allUsersPayloadSymbols = np.zeros((K, oneUserPayloadLength), dtype=complex)
    
    allUsersPacketSymbols = np.zeros((K, oneUserPacketLength), dtype=complex)
    allUsersPayloadWords = np.zeros((K, oneUserPayloadLength), dtype=int)
    allUsersSamplesFiltered = np.zeros((K, oneUserPacketLength*oversamplingRatio), dtype=complex)
    allUsersPayloadSymbols = np.zeros((K, oneUserPayloadLength), dtype=complex)
    for userIndex in range(0, K):
        #Apply a matched filter to the samples
        oneUserSamplesFiltered = signal.convolve(allUsersPacketSamples[userIndex, :], pulseShape)
        #Remove the start and end regions of the pulse shaping filter
        oneUserSamplesFiltered = oneUserSamplesFiltered[int((len(pulseShape)-1)/2):-int((len(pulseShape)-1)/2)]
        allUsersSamplesFiltered[userIndex, :] = oneUserSamplesFiltered
        
    if golayFlag:
        #The input data is oversampled by some amount
        #For now, it is always assumed to be oversampled by 2x
        #Use the Golay sequences to choose the best sampling phase, then
        #   decimate
        for userIndex in range(0, K):
            allUsersPacketSymbols[userIndex, :] = pickSamplingPhaseGolay(allUsersSamplesFiltered[userIndex, :], golaySequenceA, golaySequenceB, oversamplingRatio)
    else:
        allUsersPacketSymbols = allUsersSamplesFiltered[:, 0::oversamplingRatio]
    
    #If there is no FDE, then no equalization is required, so just take the payload as is
    channelEstimateMatrix = np.zeros((), dtype=complex)
    if not fdeFlag:
        allUsersPayloadSymbols = allUsersPacketSymbols[:, oneUserPayloadStartIndex:]
    else:
        #This is an estimate of the inter-user interference for each subcarrier
        #This is estimated by looking at each user stream when one particular user is transmitting
        #   their FDE pilots. If there is no interference, then the pilot will only appear
        #   in that user's stream, so the estimate matrix will be diagonal
        #This can also be used to equalize the channel for each user; this shows up as
        #   non-unity diagonal entries. There should be some issue with this because
        #   you have way less data than at the antennas (KxK vs. KxM), but I need to think about this again
        channelEstimateMatrix = np.zeros((fdeBlockLength, K, K), dtype=complex)
        for userIndex in range(0, K):
            #Generate the pilots (subcarrier domain) which were transmitted by each User
            oneUserFdePilotSubcarrierWords = allUsersFdePilotSubcarrierWords[userIndex, :]
            oneUserFdePilotsArray = oneUserFdePilotSubcarrierWords.reshape((numFdePilots, -1))
            
            #For each user, want to find the position which corresponds to the start of their pilots
            #Each user sends pilots in a TDD fashion
            oneUserRxFdePilotBaseStartIndex = oneUserFdePilotStartIndex + userIndex*numFdePilots*fdeBlockAndCPLength
            for pilotIndex in range(0, numFdePilots):
                #Create BPSK pilots
                oneUserTxFdePilotSubcarrierSymbols = qamMod(oneUserFdePilotsArray[pilotIndex, :], 2)
                
                oneUserRxFdePilotStartIndex = oneUserRxFdePilotBaseStartIndex + pilotIndex*fdeBlockAndCPLength
                
                #Because zero-forcing is not yet processed, a single pilot from User0 could end up
                #   in UserN's stream, so need to take look at all user streams
                allUsersRxFdePilot = allUsersPacketSymbols[:, oneUserRxFdePilotStartIndex+np.arange(0,fdeBlockAndCPLength)]
                
                for beamIndex in range(0, K):
                    #Only look at the pilot sequence after the cyclic prefix
                    #The cyclic prefix is only there to add ISI to the first few symbols of the real pilot
                    #Construct an estimate of the interuser channel for each subcarrier
                    #Because there are multiple pilots, want to add to the current channelEstimateMatrix
                    beamRxPilotTimeDomain = allUsersRxFdePilot[beamIndex, fdeCyclicPrefixLength:]
                    #Convert the time-domain interference to subcarrier domain and divide by the subcarrier domain pilot symbols
                    channelGain = 1/np.sqrt(fdeBlockLength)*np.fft.fft(beamRxPilotTimeDomain)/oneUserTxFdePilotSubcarrierSymbols
                    channelEstimateMatrix[:, beamIndex, userIndex] = channelEstimateMatrix[:, beamIndex, userIndex] + channelGain
        channelEstimateMatrix = channelEstimateMatrix/numFdePilots
        
        #Compute zero-forcing matrix based off a pseudo-inverse of the channel estimate
        #   matrix for each subcarrier
        fdeMatrices = np.zeros((fdeBlockLength, K, K), dtype=complex)
        for subcarrierIndex in range(0, fdeBlockLength):
            fdeMatrices[subcarrierIndex, :, :] = np.linalg.pinv(channelEstimateMatrix[subcarrierIndex, :, :])
            
        #Extract the data payloads from each user packet
        #Take extracted block and convert to subcarrier domain, then multiply by FDE matrix
        #Demodulate QAM signal
        allUsersPayload = allUsersPacketSymbols[:, oneUserPayloadStartIndex:]
        allUsersBlocksEqualizedTime = np.zeros((oneUserPayloadNumFdeBlocks, fdeBlockLength), dtype=complex)
        for blockIndex in range(0, oneUserPayloadNumFdeBlocks):
            #Each block has a cyclic prefix first, then a data block
            #Total size of block is fdeBlockAndCPLength
            allUsersBlockStartIndex = blockIndex*fdeBlockAndCPLength+fdeCyclicPrefixLength
            allUsersBlock = allUsersPayload[:, allUsersBlockStartIndex+np.arange(0, fdeBlockLength)]
            allUsersBlockFft = np.fft.fft(allUsersBlock, axis=1)
            allUsersBlockEqualized = np.zeros((K, fdeBlockLength), dtype=complex)
            
            for subcarrierIndex in range(0, fdeBlockLength):
                allUsersBlockEqualized[:, subcarrierIndex] = np.dot(fdeMatrices[subcarrierIndex, :, :], allUsersBlockFft[:, subcarrierIndex])
                
            allUsersBlocksEqualizedTime[blockIndex, :] = np.fft.ifft(allUsersBlockEqualized, axis=1)
            
        allUsersPayloadSymbols[userIndex, :] = allUsersBlocksEqualizedTime.flatten()
        
    #Scale constellation to have an average power of 1
    #This is used to convert the payload symbols into QAM symbols such that they can be demodulated
    normalizationScalar = modNorm(qamMod(range(0, qamOrder), qamOrder), 'avpow')
    
    for userIndex in range(0, K):
        allUsersPayloadWords[userIndex, :] = qamDemod(1/normalizationScalar*allUsersPayloadSymbols[userIndex, :], qamOrder)
    
    return allUsersPayloadWords, allUsersPayloadSymbols, allUsersPacketSymbols, channelEstimateMatrix
                
def pickSamplingPhaseGolay(oneUserPacketSymbols, golayA, golayB, oversamplingRatio):
    golaySequenceLength = len(golayA)
    
    golayA = golayA.astype(complex)
    golayB = golayB.astype(complex)
    
    #Phase 0
    seqAPhase0 = oneUserPacketSymbols[0:-2*golaySequenceLength:2]
    seqBPhase0 = oneUserPacketSymbols[2*golaySequenceLength::2]
    corrPhase0 = 1/2/golaySequenceLength*(signal.convolve(np.flip(golayA, axis=0),seqAPhase0)+signal.convolve(np.flip(golayB, axis=0),seqBPhase0))
    
    #Phase 1
    seqAPhase1 = oneUserPacketSymbols[1:-2*golaySequenceLength:2]
    seqBPhase1 = oneUserPacketSymbols[2*golaySequenceLength+1::2]
    corrPhase1 = 1/2/golaySequenceLength*(signal.convolve(np.flip(golayA, axis=0),seqAPhase1)+signal.convolve(np.flip(golayB, axis=0),seqBPhase1))
    
    peakCorrPhase0 = np.max(np.abs(corrPhase0)**2)/np.mean(np.abs(corrPhase0)**2)
    peakCorrPhase1 = np.max(np.abs(corrPhase1)**2)/np.mean(np.abs(corrPhase1)**2)
    
    if peakCorrPhase0 > peakCorrPhase1:
        oneUserCorrectlySampledSymbols = oneUserPacketSymbols[0::oversamplingRatio]
    else:
        oneUserCorrectlySampledSymbols = oneUserPacketSymbols[1::oversamplingRatio]
        
    return oneUserCorrectlySampledSymbols

def qamModulateBitStream(bitStream, qamOrder):
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
    
    qamSymbolStream = normalizationScalar*qamMod(wordStream, qamOrder)
    
    return wordStream, qamSymbolStream

def qamDemod(symbolStream, qamOrder):
    realMaxValue = int(np.sqrt(qamOrder)-1)
    
    realWordStream = np.clip(np.round((np.real(symbolStream)+realMaxValue)/2), 0, realMaxValue).astype(int)
    imagWordStream = np.clip(np.round((np.imag(symbolStream)+realMaxValue)/2), 0, realMaxValue).astype(int)
    
    wordStream = imagWordStream + (realMaxValue+1)*realWordStream
    
    return wordStream

def qamMod(wordStream, qamOrder):
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
    
    symbolStream = qamDictionary[wordStream]
    return symbolStream

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

#DO I SERIOUSLY NEED TO WRITE THIS FUCKING FUNCTION???!?!?!
def oversampleWaveform(inputSamples, oversamplingRatio, pulseShape):
    outputWaveform = np.zeros((len(inputSamples)*oversamplingRatio), dtype=complex)
    outputWaveform[0:len(outputWaveform):oversamplingRatio] = inputSamples
    
    outputWaveform = signal.convolve(outputWaveform, pulseShape)
    outputWaveform = outputWaveform[int((len(pulseShape)-1)/2):-int((len(pulseShape)-1)/2)]
    return outputWaveform