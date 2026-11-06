"""Creates the datasets (run once: python make_data.py).

data/notes.csv         synthetic notes made from templates (id,text,category,topic)
data/handwritten.csv   notes I wrote by hand in different wording (text,category)
                       category 'unsure' = notes that fit NO category
"""
import csv, random

random.seed(42)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["9 AM", "10 AM", "11:30 AM", "2 PM", "4 PM", "5:30 PM"]
DATES = ["the 5th", "the 12th", "the 18th", "the 25th", "the 28th", "the 30th"]
CITIES = ["Delhi", "Mumbai", "Hyderabad", "Chennai", "Bangalore", "Goa", "Pune"]

# topic -> (category, templates)
TOPICS = {
    "python_exam": ("study", [
        "My Python exam is on {day} at {time}.",
        "Python exam {day}, revise loops and functions",
        "Don't forget the Python test on {day} morning",
        "Revise Python OOP before the exam on {date}"]),
    "dsa_assignment": ("study", [
        "Finish DSA assignment on trees by {day}",
        "DSA lab record submission on {day}",
        "Solve 5 graph problems for DSA practice before {date}",
        "Submit the DSA assignment before {date}"]),
    "maths_exam": ("study", [
        "Maths exam on {day} at {time}, revise integration",
        "Study calculus chapters for the maths test on {date}",
        "Maths internal exam {day}, carry calculator"]),
    "electricity_bill": ("bills", [
        "Pay the electricity bill before {date}.",
        "Electricity bill due {date}, around 1200 rupees",
        "Electricity bill payment pending, pay on {day}"]),
    "rent": ("bills", [
        "Pay house rent by {date}",
        "Rent is due on {date}, transfer to the landlord",
        "Send the monthly rent payment on {day}"]),
    "credit_card": ("bills", [
        "Credit card bill due on {date}",
        "Pay the credit card payment before {date} to avoid late fee",
        "Credit card statement ready, pay by {day}"]),
    "internet_bill": ("bills", [
        "Recharge the internet broadband plan by {date}",
        "Wifi bill payment due {date}",
        "Pay the internet bill on {day}"]),
    "flight": ("travel", [
        "Flight to {city} is booked for {day} morning.",
        "Flight to {city} on {day}, reach the airport 2 hours early",
        "Check in online for the {city} flight on {day}",
        "Flight booking to {city} confirmed for {date}"]),
    "train": ("travel", [
        "Train to {city} on {day} at {time}, platform 3",
        "Book train tickets to {city} for {date}",
        "Train ticket to {city} is waitlisted for {day}"]),
    "hotel": ("travel", [
        "Hotel reservation in {city} for {day} night",
        "Book a hotel in {city} near the station for {date}",
        "Hotel check-in at {city} on {day} after {time}"]),
    "dentist": ("appointments", [
        "Dentist appointment on {day} at {time}",
        "Dentist visit {date}, clinic near the market",
        "Doctor appointment on {day} at {time}, carry old reports"]),
    "client_meeting": ("appointments", [
        "Client meeting on {day} at {time}",
        "Team meeting {day} at {time}, prepare the agenda",
        "Meeting with the manager on {day} at {time}"]),
    "interview": ("appointments", [
        "Interview scheduled on {day} at {time}",
        "Online interview {date} at {time}, test the laptop camera",
        "Interview with the hiring team on {day}, wear formal clothes"]),
    "groceries": ("shopping", [
        "Buy milk, eggs and bread from the grocery store",
        "Grocery list: rice, dal, vegetables and fruits",
        "Buy vegetables and fruits on the way back on {day}",
        "Groceries needed: cooking oil, sugar, tea powder"]),
    "gadgets": ("shopping", [
        "Buy new headphones, budget under 3000 rupees",
        "Order a phone charger and a laptop bag online",
        "Compare prices of a new keyboard before buying",
        "Buy a power bank during the sale on {date}"]),
    "clothes": ("shopping", [
        "Buy new shoes for the weekend",
        "Order two shirts and a jacket online",
        "Buy a gift and some clothes for the festival on {date}"]),
}


def fill(t):
    return t.format(day=random.choice(DAYS), time=random.choice(TIMES),
                    date=random.choice(DATES), city=random.choice(CITIES))


rows, seen = [], set()
for topic, (cat, templates) in TOPICS.items():
    tries = 0
    count = 0
    while count < 12 and tries < 400:
        tries += 1
        text = fill(random.choice(templates))
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, cat, topic))
        count += 1
random.shuffle(rows)

with open("data/notes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "text", "category", "topic"])
    for i, (text, cat, topic) in enumerate(rows):
        w.writerow([i, text, cat, topic])

# Hand-written notes in wording the templates never used.
HAND = [
    # study
    ("OS viva next Tuesday, go through deadlock and paging", "study"),
    ("Need to finish the DBMS record book before lab", "study"),
    ("Physics test tomorrow morning, revise chapter 4 formulas", "study"),
    ("Practice C pointers questions tonight", "study"),
    ("Submit the mini project report to the professor by Thursday", "study"),
    ("Chemistry practical exam on the 14th", "study"),
    # bills
    ("Gas cylinder booking amount to be paid this week", "bills"),
    ("Mobile postpaid bill is pending, pay before the 10th", "bills"),
    ("Insurance premium due on the 20th", "bills"),
    ("Water bill reminder, pay at the municipal office", "bills"),
    ("Pay college fees installment by the 15th", "bills"),
    ("EMI for the laptop gets deducted on the 5th, keep balance", "bills"),
    # travel
    ("Bus to Warangal leaves at 6 AM on Saturday", "travel"),
    ("Pack bags for the Goa trip on Friday night", "travel"),
    ("Cab to the airport booked for 4 AM", "travel"),
    ("Passport and tickets must be in the bag before the journey", "travel"),
    ("Flight to Delhi is booked for Monday morning", "travel"),
    ("Visit Ooty with friends in December, check train options", "travel"),
    # appointments
    ("Eye checkup at 5 PM on Wednesday", "appointments"),
    ("Haircut appointment on Sunday at 11", "appointments"),
    ("Bank visit to update the passbook on Monday at 10", "appointments"),
    ("Project review call with the mentor on Tuesday evening", "appointments"),
    ("Vaccination slot booked for Thursday 3 PM", "appointments"),
    ("Placement interview at 9:30 on Friday", "appointments"),
    # shopping
    ("Need to buy a notebook and pens for class", "shopping"),
    ("Pick up bananas, curd and onions", "shopping"),
    ("Buy a birthday cake and decorations for Saturday", "shopping"),
    ("Order a new water bottle and a backpack", "shopping"),
    ("Get detergent and soap from the supermarket", "shopping"),
    ("Buy a mouse and a USB cable for the laptop", "shopping"),
    # notes that fit NO category: the app should say "unsure"
    ("Call grandma this weekend", "unsure"),
    ("Idea: build a weather app with a clean design", "unsure"),
    ("Remember to water the plants every morning", "unsure"),
    ("Learn guitar chords for that old song", "unsure"),
    ("Football match with friends on Sunday evening", "unsure"),
    ("Meditate for ten minutes after waking up", "unsure"),
    ("Ravi's birthday is next week", "unsure"),
    ("Read Atomic Habits before the year ends", "unsure"),
    ("Clean the room and fold the clothes", "unsure"),
    ("Try the new biryani place near the college", "unsure"),
    ("Write a letter to my school teacher", "unsure"),
    ("Sleep early today", "unsure"),
]
with open("data/handwritten.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "category"])
    w.writerows(HAND)

print(f"notes.csv: {len(rows)} notes, handwritten.csv: {len(HAND)} notes")
