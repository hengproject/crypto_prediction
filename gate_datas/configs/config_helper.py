from __future__ import print_function
import gate_api
from gate_api.exceptions import ApiException, GateApiException

# 虚拟货币类型
raw_crypto_type_list = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "TON", "ADA", "TRX", "AVAX", "SHIB", "LINK", "DOT",
                        "BCH", "NEAR", "LEO", "LTC", "ICP", "PEPE", "UNI", "SUI", "KAS", "APT", "FET", "TAO", "RENDER",
                        "POL", "ETC", "XLM", "XMR", "STX", "IMX", "OKB", "AAVE", "FIL", "OP", "ARB", "HBAR", "CRO",
                        "WIF", "INJ", "MNT", "VET", "ATOM", "FTM", "RUNE", "GRT", "BONK", "FLOKI", "SEI", "MKR", "AR",
                        "BGB", "THETA", "TIA", "PYTH", "JUP", "HNT", "MATIC", "JASMY", "LDO", "ALGO", "ONDO", "OM",
                        "CORE", "BSV", "WLD", "BTT", "KCS", "BRETT", "NOT", "FLOW", "QNT", "POPCAT", "BEAM", "GALA",
                        "ORDI", "STRK", "CFX", "CKB", "GT", "EOS", "W", "AXS", "EGLD", "NEO", "FLR", "XTZ", "PENDLE",
                        "AKT", "XEC", "1000SATS", "SAND", "DYDX""DYDX", "RON", "ENS", "XAUt", "MANA", "MINA", "CHZ",
                        "CAKE", "NEXO", "AIOZ", "AXL", "ZRO", "SNX", "KLAY", "ROSE", "MOG", "ZK", "BOME", "LUNC", "MEW",
                        "LPT", "DEXE", "ASTR", "SUPER", "PAXG", "ZEC", "APE", "IOTA", "TFUEL", "SAFE", "FTT", "RAY",
                        "DOGS", "BLUR", "GMT", "OSMO", "GNO", "BTG", "XDC", "COMP", "TWT", "IOTX", "NFT"]

# 手动确认添加的
_include_type_list = ['SATS']


def _exclude_crypto_non_exist_spot(list_of_crypto_types):
    configuration = gate_api.Configuration(
        host="https://api.gateio.ws/api/v4"
    )
    api_client = gate_api.ApiClient(configuration)
    # Create an instance of the API class
    api_instance = gate_api.SpotApi(api_client)
    l = []
    new_list = list_of_crypto_types.copy()
    try:
        # List all currency pairs supported
        api_response = api_instance.list_currency_pairs()
        for currency_pair in api_response:
            l.append(currency_pair.base)
    except GateApiException as ex:
        print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
    except ApiException as e:
        print("Exception when calling SpotApi->list_currency_pairs: %s\n" % e)
    for each in new_list:
        if each not in l:
            new_list.remove(each)
    return new_list


def _exclude_crypto_non_exist_future(list_of_crypto_types):
    configuration = gate_api.Configuration(
        host="https://api.gateio.ws/api/v4"
    )
    new_list = list_of_crypto_types.copy()
    new_list = [f"{crypto.upper()}_USDT" for crypto in new_list]
    api_client = gate_api.ApiClient(configuration)
    # Create an instance of the API class
    api_instance = gate_api.FuturesApi(api_client)
    future_list = []

    def _get_future_datas(offset=0):
        try:
            ret = []
            # List all futures contracts
            api_response = api_instance.list_futures_contracts('usdt', limit=100, offset=offset)
            # print(api_response)
            for currency_pair in api_response:
                ret.append(currency_pair.name)
            return ret
        except GateApiException as ex:
            print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
        except ApiException as e:
            print("Exception when calling FuturesApi->list_futures_contracts: %s\n" % e)

    tmp = _get_future_datas()
    i = 0
    while tmp:
        i += 1
        future_list = future_list + tmp
        tmp = _get_future_datas(i * 100)
    print(future_list)
    for each in new_list:
        if each not in future_list:
            new_list.remove(each)
    return new_list


def get_spot_crypto_list():
    new_crypto_type_list = _exclude_crypto_non_exist_spot(raw_crypto_type_list)
    return [f"{crypto.upper()}_USDT" for crypto in new_crypto_type_list]


def get_future_crypto_list():
    new_crypto_type_list = _exclude_crypto_non_exist_future(raw_crypto_type_list)
    return new_crypto_type_list


crypto_type_spot_list_helper = get_spot_crypto_list()
crypto_type_future_list_helper = get_future_crypto_list()
