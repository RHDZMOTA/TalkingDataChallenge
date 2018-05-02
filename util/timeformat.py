import datetime as dt


def now(only_date=True):
    if only_date:
        return dt.datetime.now().strftime("%Y%m%d")
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")
