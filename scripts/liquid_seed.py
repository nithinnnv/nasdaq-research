"""A seed list of liquid Nasdaq large-caps for prototyping.

Hardcoded on purpose: pulling today's index membership would bias the
prototype toward names that are *currently* large, and the point here is to
exercise the pipeline, not to measure anything. The real universe comes from
data/universe.py plus a liquidity screen on measured dollar volume.
"""
SEED = """
AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO TSLA COST NFLX AMD PEP ADBE CSCO
TMUS INTC QCOM TXN AMGN INTU CMCSA HON AMAT BKNG ISRG VRTX ADP GILD REGN
MU LRCX PANW ADI MDLZ SBUX KLAC SNPS CDNS MELI MAR CTAS ORLY CSX ABNB
FTNT ADSK NXPI PCAR ROP MNST PAYX AEP KDP CPRT ODFL FAST EA CHTR EXC
IDXX CTSH BKR XEL VRSK CCEP DXCM ANSS ZS TTD TEAM DDOG CRWD MRVL WBD
ILMN GEHC CSGP FANG BIIB WDAY ON MCHP SMCI MSTR ARM APP AXON LIN PDD
JD BIDU NTES TCOM ZM DOCU OKTA ROKU PLTR RIVN LCID SIRI TRIP EBAY
""".split()
