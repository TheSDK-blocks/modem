# MODEM module 
# Provides helper functions for data modulation and demodulation
# From the codes of Luke Calderin cowardly stolen, modified and added by Marko Kosunen
#
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 17.11.2017 14:32
##############################################################################
import numpy as np

def pilot2ofdmsymbols(ofdmdict,pilotstream):
    #An example of an ofdmfdict
    ##ofdmdict is a dictionary describing the frame structure of the
    ##OFDM sybol
    #ofdm64dict_ngb={ 'framelen':64,'data_loc': np.r_[1:11+1, 13:25+1, 27:39+1, 41:53+1, 55:64+1]-1,
            #'pilot_loc' : np.r_[-21, -7, 7, 21] + 32, 'CPlen':16}
    pilotblocklen=len(pilot_loc)
    nsyms=int(np.ceil(len(pilotstream)/pilotblocklen))
    ##Zero-pad pilots to full blocks
    pilots=np.zeros((nsyms*pilotblocklen))
    pilots[0:len(pilotstream)]=pilotstream
    pilots=pilots.reshape((nsyms,pilotblocklen))
    return pilotsymbols

def data2ofdmsymbols(ofdmdict,datastream):
    #An example of an ofdmfdict
    ##ofdmdict is a dictionary describing the frame structure of the
    ##OFDM sybol
    #ofdm64dict_ngb={ 'framelen':64,'data_loc': np.r_[1:11+1, 13:25+1, 27:39+1, 41:53+1, 55:64+1]-1,
            #'pilot_loc' : np.r_[-21, -7, 7, 21] + 32, 'CPlen':16}
    datablocklen=len(data_loc)
    CPlen=ofdmdict['CPlen']
    nsyms=int(np.ceil(len(datastream)/datablocklen))
    #Zero-pad data to full blocks
    data=np.zeros((nsyms*datablocklen))
    data[0:len(datastream)]=datastream
    ofdmsymbols=data.reshape((nsyms,datablocklen))
    return ofdmsymbols


def ofdmMod(ofdmdict, dataframe, pilotframe):
    ##ofdmdict is a dictionary describing the frame structure of the
    framelen=ofdmdict['framelen']
    data_loc=ofdmdict['data_loc']
    pilot_loc=ofdmdict['pilot_loc']
    pilotblocklen=len(pilot_loc)
    CPlen=ofdmdict['CPlen']
    nsyms=np.min((dataframe.shape[0],pilotframe.shape[0]))

    #Equalize the number of symbols
    data=dataframe[0:nsyms,:]
    pilots=pilotframe[0:nsyms,:]
    frames=np.zeros((nsyms,framelen),dtype='complex')
    frames[:,data_loc]=dataframe
    frames[:,pilot_loc]=pilotframe
    
    #frames=np.r_['1',frames[:,32::], frames[:,0:32]]
    ofdmsyms=np.fft.ifft(frames,axis=1)*framelen
    #Add cyclic prefix
    ofdmsyms=np.r_['1',ofdmsyms[:,-CPlen::],ofdmsyms]
    test=np.fft.fft(ofdmsyms[0,16:80])/64
    
    #Maybe add Windowing here
    #then rehape to vector
    ofdmsig=ofdmsyms.reshape((1,nsyms*(framelen+CPlen)))
    return ofdmsig

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
    #Bytepad and packbits convert the bitstream to integers
    #Is this the mapping defined in standars, or does it matter? MK 
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

def modNorm(symbolStream, powerMetric):
    if powerMetric == 'avpow':
        meanPower = np.mean(np.abs(symbolStream)**2)
        normalizingFactor = np.sqrt(1/meanPower)
    
    return normalizingFactor

def nextPow2(inputValue):
    nextValue = np.ceil(np.log2(inputValue))
    return nextValue

if __name__=="__main__":
    import matplotlib.pyplot as plt
    ofdmdict=ofdm64dict_noguardband
    frame=np.zeros((100,ofdmdict['framelen']))
    frame[:,[13,17,18]]=1 #Some carriers set to constant
    datasymbols=frame[:,ofdmdict['data_loc']]
    pilotsymbols=frame[:,ofdmdict['pilot_loc']]
    modsig=ofdmMod(ofdmdict,datasymbols,pilotsymbols)
    #print(np.real(modsig)[0,:])
    plt.plot(np.real(modsig)[0,:])
    plt.show()


    
