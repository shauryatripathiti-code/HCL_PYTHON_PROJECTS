# Hotel Bookings Analysis (pandas and numpy only)
import pandas as pd
import numpy as np

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 30)
line = "=" * 80

# Number of rooms of each type (assumed, change to the real hotel's numbers)
rooms = {"Standard": 4, "Deluxe": 3, "Executive": 2, "Family Suite": 1, "Presidential Suite": 1, "Budget Single": 1}
total_rooms = sum(rooms.values())

print("\n" + line + "\nTASK 1. LOAD AND INSPECT THE BOOKING RECORDS\n" + line)
df = pd.read_csv("13_hotel_bookings.csv")
print("Rows, columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types and non-null counts:")
df.info()
print("\nSummary of numeric columns:")
print(df.describe())

print("\n" + line + "\nTASK 2. MISSING VALUES AND DUPLICATE Booking_ID RECORDS\n" + line)
missing = df.isnull().sum()
print("Missing values per column (only columns with gaps):")
print(missing[missing > 0])
print("\nDuplicate Booking_ID rows (extra copies):", df["Booking_ID"].duplicated().sum())
print("Fully identical duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates(subset="Booking_ID").reset_index(drop=True)
print("Rows after removing duplicates:", len(df))

print("\n" + line + "\nTASK 3. HANDLE MISSING Lead_Time_Days AND Room_Rate\n" + line)
# lead time = check-in date - booking date, so it can be recalculated exactly
lead = (pd.to_datetime(df["Checkin_Date"]) - pd.to_datetime(df["Booking_Date"])).dt.days
print("Lead_Time_Days: filled", df["Lead_Time_Days"].isnull().sum(), "gaps from Booking_Date and Checkin_Date.")
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(lead)
# room rate: median of the same room type (a few expensive suites would pull a mean up)
print("Room_Rate: filled", df["Room_Rate"].isnull().sum(), "gaps with the median rate of the same Room_Type.")
df["Room_Rate"] = df["Room_Rate"].fillna(df.groupby("Room_Type")["Room_Rate"].transform("median"))
print("Missing values left:", df[["Lead_Time_Days", "Room_Rate"]].isnull().sum().sum())

print("\n" + line + "\nTASK 4. CHECK-IN DATE AS DATETIME, ARRIVAL MONTH AND WEEKDAY\n" + line)
df["Checkin_Date"] = pd.to_datetime(df["Checkin_Date"])
df["Arrival_Month"] = df["Checkin_Date"].dt.month
df["Arrival_Month_Name"] = df["Checkin_Date"].dt.month_name()
df["Arrival_Weekday"] = df["Checkin_Date"].dt.day_name()
df["Year_Month"] = df["Checkin_Date"].dt.strftime("%Y-%m")
print("Checkin_Date type:", df["Checkin_Date"].dtype)
print(df[["Checkin_Date", "Arrival_Month", "Arrival_Month_Name", "Arrival_Weekday"]].head())
print("Data covers:", df["Checkin_Date"].min().date(), "to", df["Checkin_Date"].max().date())

print("\n" + line + "\nTASK 5. TOTAL BOOKINGS, ROOM NIGHTS AND CANCELLATION PERCENTAGE\n" + line)
df["Is_Canceled"] = df["Booking_Status"] == "Canceled"
kept = df[~df["Is_Canceled"]]          # non-canceled bookings
print("Total bookings:", len(df))
print("Total room nights (all bookings):", df["Nights"].sum())
print("Room nights (non-canceled bookings):", kept["Nights"].sum())
print(f"Overall cancellation rate: {df['Is_Canceled'].mean() * 100:.2f}%  ({df['Is_Canceled'].sum()} of {len(df)})")

print("\n" + line + "\nTASK 6. BOOKING REVENUE = Nights x Room_Rate (non-canceled only)\n" + line)
df["Value"] = df["Nights"] * df["Room_Rate"]
df["Revenue"] = np.where(df["Is_Canceled"], 0, df["Value"])
kept = df[~df["Is_Canceled"]]
print(df[["Booking_ID", "Nights", "Room_Rate", "Booking_Status", "Revenue"]].head(8))
print(f"\nTotal realised revenue: Rs {df['Revenue'].sum():,.0f}")

print("\n" + line + "\nTASK 7. AVERAGE ROOM RATE AND OCCUPANCY BY ROOM TYPE\n" + line)
days = (df["Checkin_Date"].max() - df["Checkin_Date"].min()).days + 1
room = kept.groupby("Room_Type").agg(Bookings=("Booking_ID", "count"), Room_Nights=("Nights", "sum"),
                                     Avg_Rate=("Room_Rate", "mean"), Median_Rate=("Room_Rate", "median"),
                                     Revenue=("Revenue", "sum"))
room["Revenue_Share_%"] = room["Revenue"] / room["Revenue"].sum() * 100
room["Rooms"] = pd.Series(rooms)
room["Occupancy_%"] = room["Room_Nights"] / (room["Rooms"] * days) * 100
room["RevPAR"] = room["Revenue"] / (room["Rooms"] * days)       # revenue per available room per day
print(f"(Occupancy uses {total_rooms} assumed rooms over {days} days)")
print(room.sort_values("Revenue", ascending=False).round(2))
print("\nRoom type x booking channel:")
print(kept.groupby(["Room_Type", "Booking_Channel"]).agg(Bookings=("Booking_ID", "count"),
      Avg_Rate=("Room_Rate", "mean"), Revenue=("Revenue", "sum")).round(0))

print("\n" + line + "\nTASK 8. BOOKING CHANNELS RANKED BY VOLUME AND REVENUE\n" + line)
ch = df.groupby("Booking_Channel")["Booking_ID"].count().to_frame("Bookings")
ch["Non_Canceled"] = kept.groupby("Booking_Channel")["Booking_ID"].count()
ch["Revenue"] = kept.groupby("Booking_Channel")["Revenue"].sum()
ch["Avg_Rate"] = kept.groupby("Booking_Channel")["Room_Rate"].mean()
ch["Volume_Rank"] = ch["Bookings"].rank(ascending=False).astype(int)
ch["Revenue_Rank"] = ch["Revenue"].rank(ascending=False).astype(int)
ch["Revenue_per_Booking"] = ch["Revenue"] / ch["Bookings"]
print(ch.sort_values("Revenue", ascending=False).round(1))

print("\n" + line + "\nTASK 14. LEAD-TIME CATEGORIES WITH pd.cut()\n" + line)
labels = ["Last-Minute (0-3d)", "Short-Term (4-14d)", "Medium (15-60d)", "Advance (60d+)"]
df["Lead_Band"] = pd.cut(df["Lead_Time_Days"], bins=[-1, 3, 14, 60, 1000], labels=labels)
print(df["Lead_Band"].value_counts(sort=False).to_frame("Bookings"))

print("\n" + line + "\nTASK 9. CANCELLATION RATE BY CHANNEL AND LEAD-TIME BAND\n" + line)
cc = df.groupby("Booking_Channel").agg(Bookings=("Booking_ID", "count"), Canceled=("Is_Canceled", "sum"),
                                       Cancel_Rate_pct=("Is_Canceled", "mean"))
cc["Cancel_Rate_pct"] = cc["Cancel_Rate_pct"] * 100
cc["Lost_Revenue_INR"] = df[df["Is_Canceled"]].groupby("Booking_Channel")["Value"].sum()
cc = cc.fillna(0).sort_values("Cancel_Rate_pct", ascending=False).round(1)
print(cc)
cb = df.groupby("Lead_Band", observed=True).agg(Bookings=("Booking_ID", "count"), Cancel_Rate_pct=("Is_Canceled", "mean"))
cb["Cancel_Rate_pct"] = (cb["Cancel_Rate_pct"] * 100).round(1)
print("\nBy lead-time band:")
print(cb)
print("\nCancellation % - channel x lead band (blank = no bookings):")
print((pd.pivot_table(df, index="Booking_Channel", columns="Lead_Band", values="Is_Canceled", observed=True) * 100).round(1))

print("\n" + line + "\nTASK 10. TOP 3 PEAK REVENUE MONTHS\n" + line)
month_revenue = df.groupby("Year_Month")["Revenue"].sum()
top3 = month_revenue.sort_values(ascending=False).head(3)
print(top3)

print("\n" + line + "\nTASK 11. AVERAGE STAY: LEISURE vs BUSINESS\n" + line)
stay = kept.groupby("Travel_Purpose")["Nights"].agg(["count", "mean", "median"]).round(2)
print(stay.sort_values("mean", ascending=False))
print("\n(leisure vs business only)")
print(stay.loc[["Leisure", "Business"]])

print("\n" + line + "\nTASK 12. NUMPY MEAN, MEDIAN AND STANDARD DEVIATION OF ROOM RATES\n" + line)
rates = kept["Room_Rate"].to_numpy()
daily = kept.groupby("Checkin_Date")["Room_Rate"].mean().to_numpy()      # average rate for each arrival day
print(f"Per booking  mean={np.mean(rates):,.1f}  median={np.median(rates):,.1f}  std={np.std(rates):,.1f}")
print(f"Daily average ({len(daily)} days)  mean={np.mean(daily):,.1f}  median={np.median(daily):,.1f}  std={np.std(daily):,.1f}")

print("\n" + line + "\nTASK 13. UNUSUALLY HIGH BOOKING VALUES (IQR METHOD)\n" + line)
q1, q3 = np.percentile(kept["Revenue"], [25, 75])
iqr = q3 - q1
limit = q3 + 1.5 * iqr
high = kept[kept["Revenue"] > limit].sort_values("Revenue", ascending=False)
share = high["Revenue"].sum() / kept["Revenue"].sum() * 100
print(f"Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}  upper fence = Q3 + 1.5 x IQR = {limit:,.0f}")
print(f"{len(high)} bookings ({len(high) / len(kept) * 100:.1f}%) are above the fence and make {share:.1f}% of revenue.")
print("\nBy room type:")
print(high["Room_Type"].value_counts().to_frame("Outlier_Bookings"))
print("\nTop 8 bookings:")
print(high[["Booking_ID", "Room_Type", "Booking_Channel", "Nights", "Room_Rate", "Revenue"]].head(8))

print("\n" + line + "\nMONTHLY OCCUPANCY TREND\n" + line)
nights = kept.groupby("Year_Month")["Nights"].sum()
days_in_month = pd.to_datetime(nights.index + "-01").days_in_month.to_numpy()
occ = nights.to_frame("Room_Nights_Sold")
occ["Occupancy_pct"] = (nights.to_numpy() / (days_in_month * total_rooms) * 100).round(1)
occ["Revenue"] = month_revenue
print(occ)
print("\nOccupancy as a simple text chart (each # = 2 percentage points):")
for month, pct in occ["Occupancy_pct"].items():
    print(f"{month}  {pct:5.1f}%  {'#' * int(round(pct / 2))}")
week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
arrivals = kept.groupby("Arrival_Weekday").size().reindex(week).to_frame("Arrivals")
arrivals["Share_pct"] = (arrivals["Arrivals"] / arrivals["Arrivals"].sum() * 100).round(1)
print("\nArrivals by weekday (non-canceled) - drives front-desk staffing:")
print(arrivals)

print("\n" + line + "\nTASK 15. FIVE REVENUE AND OPERATIONS RECOMMENDATIONS\n" + line)
low = ", ".join(occ["Occupancy_pct"].nsmallest(3).index)
peak = ", ".join(top3.index)
ota_loss = cc[cc.index.str.startswith("OTA")]["Lost_Revenue_INR"].sum() / cc["Lost_Revenue_INR"].sum() * 100
busy = arrivals["Arrivals"].nlargest(2)
revpar = room["RevPAR"]
print(f"\n1. Price by season and fill the quiet months. The weakest months are {low}; the best revenue months are {peak}.")
print("   Raise rates in peak months; in quiet months run early-bird, family and long-stay offers.")
print(f"\n2. Reduce OTA cancellation losses. {cc.index[0]} cancels {cc['Cancel_Rate_pct'].iloc[0]}% of bookings against")
print(f"   {cc.loc['Direct Website', 'Cancel_Rate_pct']}% for the Direct Website, and OTAs carry {ota_loss:.0f}% of all cancelled value.")
print("   Offer a cheaper non-refundable OTA rate and give perks (late check-out, breakfast) for booking direct.")
print(f"\n3. Ask for deposits on long-lead bookings. Last-minute bookings cancel {cb['Cancel_Rate_pct'].iloc[0]}% of the time but")
print(f"   advance bookings cancel {cb['Cancel_Rate_pct'].iloc[-1]}%. Take a card guarantee when the lead time is over 14 days.")
print(f"\n4. Staff the front desk to the arrival pattern. {busy.index[0]} and {busy.index[1]} are the busiest arrival days")
print(f"   ({busy.iloc[0]} and {busy.iloc[1]} arrivals); {arrivals['Arrivals'].idxmin()} is the quietest. Business guests stay only")
print(f"   {stay.loc['Business', 'mean']} nights (leisure {stay.loc['Leisure', 'mean']}), so offer them express check-out.")
print(f"\n5. Protect high-value stays and fix weak room types. {len(high)} bookings above Rs {limit:,.0f} bring {share:.0f}% of revenue:")
print(f"   promote suites to direct and corporate guests. {revpar.idxmax()} earns the most per room per day and")
print(f"   {revpar.idxmin()} the least, so review its price.")

# Save the cleaned data (with the new columns) as the single CSV output
df.to_csv("hotel_bookings_cleaned.csv", index=False)
print("\nSaved the cleaned data to hotel_bookings_cleaned.csv")