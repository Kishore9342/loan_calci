def calculate_compound_interest(principal, rate, times_compounded, years):
    amount = principal * (1 + rate/(100 * times_compounded))**(times_compounded * years)
    interest = amount - principal
    return amount, interest

def main():
    print("Compound Interest Calculator")
    
    principal = float(input("Enter the principal amount: "))
    rate = float(input("Enter the annual interest rate (in %): "))
    times_compounded = int(input("Times interest compounded per year: "))
    years = int(input("Number of years: "))

    amount, interest = calculate_compound_interest(principal, rate, times_compounded, years)
    
    print(f"\nAfter {years} years, the investment will be worth: ${amount:.2f}")
    print(f"Total interest earned: ${interest:.2f}")

if __name__ == "__main__":
    main()
