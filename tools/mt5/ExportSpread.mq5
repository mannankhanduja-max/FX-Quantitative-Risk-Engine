//+------------------------------------------------------------------+
//|  ExportSpread.mq5                                                 |
//|                                                                   |
//|  Dumps per-bar OHLC plus the BROKER'S OWN SPREAD to CSV, so a     |
//|  backtest can be costed at what this account actually charges     |
//|  rather than at an ECN aggregate.                                 |
//|                                                                   |
//|  WHY THIS EXISTS                                                  |
//|  ---------------                                                  |
//|  The study behind this repository costs every trade from          |
//|  Dukascopy's measured ask-minus-bid. That is an ECN aggregate and |
//|  is NOT what an MT5 retail account pays: a raw/ECN account adds   |
//|  commission on top, and a standard account marks the spread up    |
//|  instead. The difference decides several results outright, so it  |
//|  is worth measuring rather than assuming.                         |
//|                                                                   |
//|  MqlRates.spread is the spread in POINTS at each bar's close,     |
//|  recorded by the terminal. It is the broker's own number.         |
//|                                                                   |
//|  HOW TO RUN                                                       |
//|  ----------                                                       |
//|   1  MetaEditor -> open this file -> Compile (F7)                 |
//|   2  In MT5, open a chart of the symbol you want                  |
//|   3  Drag "ExportSpread" from Navigator onto that chart           |
//|   4  Set Inputs (below), press OK                                 |
//|   5  The CSV lands in  <terminal data folder>/MQL5/Files/         |
//|      File -> Open Data Folder  finds it                           |
//|                                                                   |
//|  IMPORTANT: scroll the chart back to the start date first, or     |
//|  press Home a few times. MT5 only has bars it has downloaded,     |
//|  and this script cannot export what the terminal has not fetched. |
//|  The script reports how many bars it actually got.                |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input datetime InpFrom      = D'2024.09.22 00:00';
input datetime InpTo        = D'2026.09.22 00:00';
input ENUM_TIMEFRAMES InpTF = PERIOD_M5;
input string   InpSuffix    = "";     // appended to the filename

//+------------------------------------------------------------------+
int OnStart()
  {
   string sym = _Symbol;

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int got = CopyRates(sym, InpTF, InpFrom, InpTo, rates);

   if(got <= 0)
     {
      PrintFormat("ExportSpread: no bars for %s. Error %d. "
                  "Scroll the chart back to %s and try again - MT5 only "
                  "has what it has downloaded.",
                  sym, GetLastError(), TimeToString(InpFrom));
      return(INIT_FAILED);
     }

   // Point size turns the integer spread into a price difference.
   double point  = SymbolInfoDouble(sym, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

   string clean = sym;
   StringReplace(clean, "/", "");
   StringReplace(clean, ".", "");
   string tf   = EnumToString(InpTF);
   StringReplace(tf, "PERIOD_", "");
   string name = StringFormat("mt5_%s_%s%s.csv", clean, tf, InpSuffix);

   int fh = FileOpen(name, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
     {
      PrintFormat("ExportSpread: cannot write %s. Error %d", name, GetLastError());
      return(INIT_FAILED);
     }

   // spread_points is the broker's spread at the bar close.
   // spread_price = spread_points * point, in quote currency.
   // The loader converts that to basis points of the mid.
   FileWrite(fh, "Datetime", "Open", "High", "Low", "Close",
             "TickVolume", "SpreadPoints", "SpreadPrice");

   for(int i = 0; i < got; i++)
     {
      double sp_price = rates[i].spread * point;
      FileWrite(fh,
                TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                DoubleToString(rates[i].open,  digits),
                DoubleToString(rates[i].high,  digits),
                DoubleToString(rates[i].low,   digits),
                DoubleToString(rates[i].close, digits),
                (long)rates[i].tick_volume,
                (int)rates[i].spread,
                DoubleToString(sp_price, digits + 2));
     }
   FileClose(fh);

   // A sanity line, because a silently short export is the easy
   // failure here - the terminal simply had not downloaded the rest.
   double med_pts = 0;
   int    n_zero  = 0;
   double acc[];
   ArrayResize(acc, got);
   for(int i = 0; i < got; i++)
     {
      acc[i] = (double)rates[i].spread;
      if(rates[i].spread == 0) n_zero++;
     }
   ArraySort(acc);
   med_pts = acc[got / 2];

   PrintFormat("ExportSpread: %s  %d bars  %s -> %s",
               name, got,
               TimeToString(rates[0].time),
               TimeToString(rates[got - 1].time));
   PrintFormat("  median spread %.0f points = %.5f price = %.3f bp of mid",
               med_pts, med_pts * point,
               med_pts * point / rates[got / 2].close * 10000.0);
   if(n_zero > got / 20)
      PrintFormat("  WARNING: %d bars (%.1f%%) report ZERO spread. Some "
                  "brokers do not record it on history bars - in that case "
                  "this export cannot be used for costing.",
                  n_zero, 100.0 * n_zero / got);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
