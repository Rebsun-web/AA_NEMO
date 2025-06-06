import holidays


class HolidayChecker:
    """Check for international holidays automatically"""

    def __init__(self):
        self.holiday_calendars = {
            'holiday_de': self.get_german_holidays,
            'holiday_be': self.get_belgian_holidays,
            'holiday_fr': self.get_french_holidays,
            'holiday_it': self.get_italian_holidays,
            'holiday_gb': self.get_uk_holidays
        }

    def check_holidays_for_date(self, date):
        """Check all international holidays for a given date"""
        holidays = {}
        for country, checker in self.holiday_calendars.items():
            holidays[country] = checker(date)

        holidays['holiday_all'] = any(holidays.values())
        return holidays

    def get_german_holidays(self, date):
        """Check German holidays - you can integrate with holidays library"""
        de_holidays = holidays.Germany(years=date.year)
        return date in de_holidays

    def get_belgian_holidays(self, date):
        be_holidays = holidays.Belgium(years=date.year)
        return date in be_holidays

    def get_french_holidays(self, date):
        fr_holidays = holidays.France(years=date.year)
        return date in fr_holidays

    def get_italian_holidays(self, date):
        it_holidays = holidays.Italy(years=date.year)
        return date in it_holidays

    def get_uk_holidays(self, date):
        uk_holidays = holidays.UK(years=date.year)
        return date in uk_holidays
