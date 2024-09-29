from __future__ import print_function
import gate_api
from gate_api.exceptions import ApiException, GateApiException
from gate_datas.configs.configs import crypto_type_list
# Defining the host is optional and defaults to https://api.gateio.ws/api/v4
# See configuration.py for a list of all supported configuration parameters.
configuration = gate_api.Configuration(
    host = "https://api.gateio.ws/api/v4"
)

api_client = gate_api.ApiClient(configuration)
# Create an instance of the API class
api_instance = gate_api.SpotApi(api_client)
l = []

try:
    # List all currency pairs supported
    api_response = api_instance.list_currency_pairs()
    for currency_pair in api_response:
        l.append(currency_pair.base)
except GateApiException as ex:
    print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
except ApiException as e:
    print("Exception when calling SpotApi->list_currency_pairs: %s\n" % e)

for each in crypto_type_list:
    if each not in l:
        print(each)

#
# c_list = ["BTC","ETH","BNB","SOL","XRP","DOGE","TON","ADA","TRX","AVAX","SHIB","LINK","DOT","BCH","NEAR","LEO","LTC","ICP","PEPE","UNI","SUI","KAS","APT","FET","TAO","RENDER","POL","ETC","XLM","XMR","STX","IMX","OKB","AAVE","FIL","OP","ARB","HBAR","CRO","WIF","INJ","MNT","VET","ATOM","FTM","RUNE","GRT","BONK","FLOKI","SEI","MKR","AR","BGB","THETA","TIA","PYTH","JUP","HNT","MATIC","JASMY","LDO","ALGO","ONDO","OM","CORE","BSV","WLD","BTT","KCS","BRETT","NOT","FLOW","QNT","POPCAT","BEAM","GALA","ORDI","STRK","CFX","CKB","GT","EOS","W","AXS","EGLD","NEO","FLR","XTZ","PENDLE","AKT","XEC","1000SATS","SAND","DYDX""DYDX","RON","ENS","XAUt","MANA","MINA","CHZ","CAKE","NEXO","AIOZ","AXL","ZRO","SNX","KLAY","ROSE","MOG","ZK","BOME","LUNC","MEW","LPT","DEXE","ASTR","SUPER","PAXG","ZEC","APE","IOTA","TFUEL","SAFE","FTT","RAY","DOGS","BLUR","GMT","OSMO","GNO","BTG","XDC","COMP","TWT","IOTX","NFT"]

