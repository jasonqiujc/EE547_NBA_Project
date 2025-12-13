// Maps all API abbreviations → the 3-letter standard version you want
const teamAbbrevMap = {
  ATL: "ATL",
  BOS: "BOS",
  BKN: "BKN",
  CHA: "CHA",
  CHI: "CHI",
  CLE: "CLE",
  DAL: "DAL",
  DEN: "DEN",
  DET: "DET",

  GS:  "GSW",   // <- API sends GS
  GSW: "GSW",

  HOU: "HOU",
  IND: "IND",
  LAC: "LAC",
  LAL: "LAL",
  MEM: "MEM",
  MIA: "MIA",
  MIL: "MIL",
  MIN: "MIN",

  NO: "NOP",   // <- API sometimes sends NO
  NOP: "NOP",

  NY: "NYK",   // <- API may send NY
  NYK: "NYK",

  OKC: "OKC",
  ORL: "ORL",
  PHI: "PHI",

  PHX: "PHX",
  PHO: "PHX",  // <- two versions but normalize to PHX

  POR: "POR",

  SA: "SAC",   // <- API incorrectly gives SA for Kings sometimes
  SAC: "SAC",

  SAS: "SAS",  // Spurs

  TOR: "TOR",
  UTA: "UTA",
  WAS: "WAS",
};

export default teamAbbrevMap;
