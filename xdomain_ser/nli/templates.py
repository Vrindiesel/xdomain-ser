# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Slot-to-template converter for NLI-based semantic accuracy evaluation.

Converts each ``(slot_name, value)`` pair into a simple declarative
sentence that can be used as an NLI hypothesis. Templates are designed
to be unambiguous for RoBERTa-MNLI entailment checking.

Covers all 133 gold MR slot names from the evaluation data across:

* E2E NLG Challenge (8 slots)
* ViGGO video games (13 slots)
* RNNLG hotel / laptop / restaurant / TV (12 / 20 / 12 / 16 slots)
* Taskmaster auto_repair / coffee / movie / pizza / restaurant_reservation /
  uber_lyft
"""


# --- Boolean / enum slot handlers -------------------------------------------

# Slots whose values are boolean-like (yes/no, has_X/no_X, etc.)
# Map: slot_name -> (positive_template, negative_template)
# Where positive matches "yes"/"has_X"/"X_allowed"/etc.

_BOOL_POSITIVE = {
    "yes", "true", "1", "has_internet", "accepts_credit_cards",
    "pets_allowed", "has_usb_port", "on_steam", "business_oriented",
    "family-friendly", "child_friendly",
}
_BOOL_NEGATIVE = {
    "no", "false", "0", "no_internet", "no_credit_cards",
    "no_pets_allowed", "no_usb_port", "not_on_steam",
    "not_business_oriented", "not-family-friendly", "not_child_friendly",
}
_DONTCARE = {"dontcare", "dont_care", "none"}


def _is_bool_positive(value):
    return value.lower().strip() in _BOOL_POSITIVE


def _is_bool_negative(value):
    return value.lower().strip() in _BOOL_NEGATIVE


def _is_dontcare(value):
    return value.lower().strip() in _DONTCARE


# --- Slot template definitions -----------------------------------------------

# Each entry is either:
#   str: a template with {value} placeholder
#   callable: function(value) -> str (for boolean/conditional slots)

def _bool_template(pos_text, neg_text):
    """Create a boolean template handler."""
    def handler(value):
        if _is_dontcare(value):
            return None  # skip dontcare slots
        if _is_bool_positive(value):
            return pos_text
        if _is_bool_negative(value):
            return neg_text
        # fallback: treat as description
        return pos_text
    return handler


SLOT_TEMPLATES = {
    # === E2E NLG Challenge (LoRA-internal slot names) ===
    "name": "The name is {value}.",
    "venue_type": "It is a {value}.",
    "cuisine_type": "The food type is {value}.",
    "priceRange": "The price range is {value}.",
    "customerRating": "The customer rating is {value}.",
    "area_zone": "It is in the {value} area.",
    "family_suitability": lambda v: (
        "It is family-friendly." if v == "family-friendly"
        else "It is not family-friendly."
    ),
    "nearby_landmark": "It is near {value}.",

    # === E2E NLG Challenge (E2E-native slot names) ===
    "eatType": "It is a {value}.",
    "food": "The food type is {value}.",
    "area": "It is in the {value} area.",
    "familyFriendly": _bool_template(
        "It is family-friendly.", "It is not family-friendly."
    ),
    "near": "It is near {value}.",

    # === ViGGO (video games) ===
    "developer": "The developer is {value}.",
    "esrb_rating": "The ESRB rating is {value}.",
    "exp_release_date": "The expected release date is {value}.",
    "genres": "The genre is {value}.",
    "multiplayer_mode": lambda v: (
        "It has multiplayer." if v == "multiplayer"
        else "It is single-player."
    ),
    "pc_os_support": lambda v: {
        "Linux": "It is available on Linux.",
        "macOS": "It is available on macOS.",
        "not_released_on_Linux": "It is not available on Linux.",
        "not_released_on_macOS": "It is not available on macOS.",
    }.get(v, f"The PC OS support is {v}."),
    "perspective": "The perspective is {value}.",
    "platforms": "It is available on {value}.",
    "released": "It was released in {value}.",
    "review_rating": "The review rating is {value}.",
    "specifier": "The game is {value}.",
    "steam_availability": lambda v: (
        "It is available on Steam." if v == "on_steam"
        else "It is not available on Steam."
    ),

    # === RNNLG Hotel (hotel_ prefix) ===
    "hotel_name": "The hotel name is {value}.",
    "hotel_type": "The venue type is {value}.",
    "hotel_price_range": "The price range is {value}.",
    "hotel_address": "The address is {value}.",
    "hotel_postcode": "The postcode is {value}.",
    "hotel_area": "It is in the {value} area.",
    "hotel_near": "It is near {value}.",
    "hotel_phone": "The phone number is {value}.",
    "hotel_count": "There are {value} results.",
    "hotel_has_internet": _bool_template(
        "It has internet.", "It does not have internet."
    ),
    "hotel_accepts_credit_cards": _bool_template(
        "It accepts credit cards.", "It does not accept credit cards."
    ),
    "hotel_pets_allowed": _bool_template(
        "Pets are allowed.", "Pets are not allowed."
    ),

    # === RNNLG Laptop (laptop_ prefix) ===
    "laptop_name": "The laptop name is {value}.",
    "laptop_type": "The device type is {value}.",
    "laptop_family": "The laptop family is {value}.",
    "laptop_price": "The price is {value}.",
    "laptop_price_range": "The price range is {value}.",
    "laptop_battery": "The battery life is {value}.",
    "laptop_battery_rating": "The battery rating is {value}.",
    "laptop_design": "The design is {value}.",
    "laptop_dimension": "The dimension is {value}.",
    "laptop_drive": "The drive size is {value}.",
    "laptop_drive_range": "The drive range is {value}.",
    "laptop_memory": "The memory is {value}.",
    "laptop_platform": "The platform is {value}.",
    "laptop_processor": "The processor is {value}.",
    "laptop_utility": "The utility is {value}.",
    "laptop_warranty": "The warranty is {value}.",
    "laptop_weight": "The weight is {value}.",
    "laptop_weight_range": "The weight range is {value}.",
    "laptop_count": "There are {value} results.",
    "laptop_is_for_business_computing": _bool_template(
        "It is for business computing.",
        "It is not for business computing."
    ),

    # === RNNLG Restaurant (restaurant_ prefix) ===
    "restaurant_name": "The restaurant name is {value}.",
    "restaurant_type": "The venue type is {value}.",
    "restaurant_food": "The food type is {value}.",
    "restaurant_price_range": "The price range is {value}.",
    "restaurant_price": "The price is {value}.",
    "restaurant_address": "The address is {value}.",
    "restaurant_postcode": "The postcode is {value}.",
    "restaurant_area": "It is in the {value} area.",
    "restaurant_near": "It is near {value}.",
    "restaurant_phone": "The phone number is {value}.",
    "restaurant_count": "There are {value} results.",
    "restaurant_good_for_meal": "It is good for {value}.",
    "restaurant_kidsallowed": lambda v: (
        "Kids are allowed." if v.lower() in ("yes", "true")
        else "Kids are not allowed." if v.lower() in ("no", "false")
        else None  # 'none' means dontcare
    ),

    # === RNNLG TV (tv_ prefix) ===
    "tv_name": "The TV name is {value}.",
    "tv_type": "The TV type is {value}.",
    "tv_family": "The TV family is {value}.",
    "tv_price": "The price is {value}.",
    "tv_price_range": "The price range is {value}.",
    "tv_resolution": "The resolution is {value}.",
    "tv_screen_size": "The screen size is {value}.",
    "tv_screen_size_range": "The screen size range is {value}.",
    "tv_eco_rating": "The eco rating is {value}.",
    "tv_hdmi_port": "It has {value} HDMI ports.",
    "tv_power_consumption": "The power consumption is {value}.",
    "tv_accessories": "The accessories include {value}.",
    "tv_audio": "The audio is {value}.",
    "tv_color": "The color is {value}.",
    "tv_count": "There are {value} results.",
    "tv_has_usb_port": _bool_template(
        "It has a USB port.", "It does not have a USB port."
    ),

    # === Taskmaster: Auto Repair (auto_repair. prefix) ===
    "auto_repair.name.store": "The auto shop name is {value}.",
    "auto_repair.name.customer": "The customer name is {value}.",
    "auto_repair.name.vehicle": "The vehicle is {value}.",
    "auto_repair.date.appt": "The appointment date is {value}.",
    "auto_repair.time.appt": "The appointment time is {value}.",
    "auto_repair.reason.appt": "The reason for the appointment is {value}.",
    "auto_repair.year.vehicle": "The vehicle year is {value}.",
    "auto_repair.location.store": "The shop location is {value}.",
    "auto_repair.appointment": "The appointment status is {value}.",

    # === Taskmaster: Coffee Ordering (coffee_ordering. prefix) ===
    "coffee_ordering.name.drink": "The drink is {value}.",
    "coffee_ordering.size.drink": "The drink size is {value}.",
    "coffee_ordering.num.drink": "The number of drinks is {value}.",
    "coffee_ordering.type.milk": "The milk type is {value}.",
    "coffee_ordering.preference": "The preference is {value}.",
    "coffee_ordering.location.store": "The store location is {value}.",
    "coffee_ordering.coffee_order": "The order status is {value}.",

    # === Taskmaster: Movie Ticket (movie_ticket. prefix) ===
    "movie_ticket.name.movie": "The movie is {value}.",
    "movie_ticket.name.theater": "The theater is {value}.",
    "movie_ticket.num.tickets": "The number of tickets is {value}.",
    "movie_ticket.time.start": "The showtime is {value}.",
    "movie_ticket.time.end": "The show ends at {value}.",
    "movie_ticket.price.ticket": "The ticket price is {value}.",
    "movie_ticket.type.screening": "The screening type is {value}.",
    "movie_ticket.location.theater": "The theater location is {value}.",
    "movie_ticket.ticket_booking": "The ticket booking status is {value}.",

    # === Taskmaster: Pizza Ordering (pizza_ordering. prefix) ===
    "pizza_ordering.name.store": "The pizza store is {value}.",
    "pizza_ordering.name.pizza": "The pizza is {value}.",
    "pizza_ordering.size.pizza": "The pizza size is {value}.",
    "pizza_ordering.type.crust": "The crust type is {value}.",
    "pizza_ordering.type.topping": "The topping is {value}.",
    "pizza_ordering.preference": "The preference is {value}.",
    "pizza_ordering.location.store": "The store location is {value}.",
    "pizza_ordering.pizza_order": "The order status is {value}.",
    "pizza_ordering.pizza_ordering": "The pizza order is {value}.",

    # === Taskmaster: Restaurant Reservation (restaurant_reservation. prefix) ===
    "restaurant_reservation.name.restaurant": "The restaurant name is {value}.",
    "restaurant_reservation.name.reservation": "The reservation name is {value}.",
    "restaurant_reservation.num.guests": "The number of guests is {value}.",
    "restaurant_reservation.time.reservation": "The reservation time is {value}.",
    "restaurant_reservation.type.seating": "The seating type is {value}.",
    "restaurant_reservation.location.restaurant": "The restaurant location is {value}.",
    "restaurant_reservation.reservation": "The reservation status is {value}.",
    "restaurant_reservation.restaurant_reservation": "The reservation is for {value}.",

    # === Taskmaster: Uber/Lyft (uber_lyft. prefix) ===
    "uber_lyft.location.from": "The pickup location is {value}.",
    "uber_lyft.location.to": "The dropoff location is {value}.",
    "uber_lyft.type.ride": "The ride type is {value}.",
    "uber_lyft.num.people": "The number of passengers is {value}.",
    "uber_lyft.price.estimate": "The estimated price is {value}.",
    "uber_lyft.duration.estimate": "The estimated duration is {value}.",
    "uber_lyft.time.pickup": "The pickup time is {value}.",
    "uber_lyft.time.dropoff": "The dropoff time is {value}.",
    "uber_lyft.ride_booking": "The ride booking status is {value}.",
    "uber_lyft.uber_lyft": "The ride service is {value}.",
}


def _make_readable_name(slot_name):
    """Convert a slot name to a readable English phrase."""
    # Strip domain prefixes (e.g., "auto_repair.", "coffee_ordering.")
    prefixes = [
        "auto_repair.", "coffee_ordering.", "movie_ticket.",
        "pizza_ordering.", "restaurant_reservation.", "uber_lyft.",
    ]
    name = slot_name
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Strip domain prefixes for RNNLG (e.g., "hotel_", "laptop_")
    rnnlg_prefixes = ["hotel_", "laptop_", "restaurant_", "tv_"]
    for prefix in rnnlg_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Convert separators to spaces
    name = name.replace(".", " ").replace("_", " ")
    return name


def slot_value_to_template(slot_name, value):
    """
    Convert a (slot_name, value) pair into a natural language template sentence.

    Returns:
        str or None: The template sentence, or None if the slot should be skipped
                     (e.g., dontcare values).
    """
    # Skip dontcare values
    if _is_dontcare(value):
        return None

    template = SLOT_TEMPLATES.get(slot_name)

    if template is not None:
        if callable(template):
            return template(value)
        return template.format(value=value)

    # Generic fallback: use readable slot name
    readable = _make_readable_name(slot_name)
    return f"The {readable} is {value}."


def get_all_registered_slots():
    """Return the set of all slot names with explicit templates."""
    return set(SLOT_TEMPLATES.keys())
