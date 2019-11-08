import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Listings:
    def __init__(self, filename = 'listings.csv', DATT = True):
        self.readListings(filename)
        if DATT:
            self.dropCols()
            self.cleaner()
    
    def readListings(self, filename):

        # suppress low_memory warning--file isn't large enough for
        # any significant processing impact.
        self.df = pd.read_csv(filename, low_memory = False)
        self.df.set_index('id')
        self.columns = self.df.columns.values.tolist()

    def nullVals(self, column):
        return self.df[column].isna().sum()

    def fillNulls(self, column, newNull):
        self.df[column] = self.df[column].fillna(newNull)

    def dollarToFloat(self, column):
        if type(self.df[column][0]) != str:
            print("Data is not a string")
        else:
            self.df[column] = self.df[column].str.replace('$', '').str.replace(',','').astype(float)

    def dropCols(self, toDrop = ['listing_url', 'scrape_id', 'last_scraped', 'name',
            'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_since',
            'host_location', 'host_acceptance_rate', 'host_neighbourhood',
            'host_listings_count', 'host_total_listings_count',
            'host_url', 'host_name', 'host_thumbnail_url', 'host_picture_url',
            'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
            'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code',
            'country', 'is_location_exact',
            'weekly_price', 'monthly_price', 'square_feet',
            'minimum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
            'maximum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
            'calendar_updated', 'calendar_last_scraped',
            'number_of_reviews_ltm', 'first_review', 'last_review',
            'jurisdiction_names']):
        self.df = self.df.drop(toDrop, axis = 1)
        self.columns = self.df.columns.values.tolist()

    def convertTextToBinary(self):
        '''Simply checks if values have been filled out such as "experiences_offered,"
        and converts to a binary'''
        toConvert = ['summary', 'space', 'description', 'experiences_offered',
                     'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',
                     'house_rules', 'thumbnail_url', 'host_about', 'license']

    def cleaner(self):

        #convert property types to an index
        self.property_types = self.df.property_type.unique().tolist()
        self.property_types_index = {}
        for i in range(len(self.property_types)):
            self.property_types_index[self.property_types[i]] = i
        self.df.property_type = self.df.property_type.apply(lambda x: self.property_types_index[x])

        #convert room types to an index
        self.room_types = self.df.room_type.unique().tolist()
        self.room_types_index = {}
        for i in range(len(self.room_types)):
            self.room_types_index[self.room_types[i]] = i
        self.df.room_type = self.df.room_type.apply(lambda x: self.room_types_index[x])

        #convert bed types to an index
        self.bed_types = self.df.bed_type.unique().tolist()
        self.bed_types_index = {}
        for i in range(len(self.bed_types)):
            self.bed_types_index[self.bed_types[i]] = i
        self.df.bed_type = self.df.bed_type.apply(lambda x: self.bed_types_index[x])

        #convert cancellation policy to an index
        self.cancellation_policies = self.df.cancellation_policy.unique().tolist()
        self.cancellation_policies_index = {}
        for i in range(len(self.cancellation_policies)):
            self.cancellation_policies_index[self.cancellation_policies[i]] = i
        self.df.cancellation_policy = self.df.cancellation_policy.apply(lambda x: self.cancellation_policies_index[x])

        #convert response time to an index
        self.response_times = self.df.host_response_time.unique().tolist()
        self.response_times_index = {}
        for i in range(len(self.response_times)):
            self.response_times_index[self.response_times[i]] = i
        self.df.host_response_time = self.df.host_response_time.apply(lambda x: self.response_times_index[x])

        #convert binary values to 0 or 1
        self.df.host_is_superhost = self.df.host_is_superhost.apply(lambda x: x == 't')
        self.df.host_has_profile_pic = self.df.host_has_profile_pic.apply(lambda x: x == 't')
        self.df.host_identity_verified = self.df.host_identity_verified.apply(lambda x: x == 't')
        self.df.has_availability = self.df.has_availability.apply(lambda x: x == 't')
        self.df.availability_30 = self.df.availability_30.apply(lambda x: x == 't')
        self.df.availability_60 = self.df.availability_60.apply(lambda x: x == 't')
        self.df.availability_90 = self.df.availability_90.apply(lambda x: x == 't')
        self.df.availability_365 = self.df.availability_365.apply(lambda x: x == 't')
        self.df.requires_license = self.df.requires_license.apply(lambda x: x == 't')
        self.df.instant_bookable = self.df.instant_bookable.apply(lambda x: x == 't')
        self.df.is_business_travel_ready = self.df.is_business_travel_ready.apply(lambda x: x == 't')
        self.df.require_guest_profile_picture = self.df.require_guest_profile_picture.apply(lambda x: x == 't')
        self.df.require_guest_phone_verification = self.df.require_guest_phone_verification.apply(lambda x: x == 't')

        #convert dollars to floats
        self.dollarToFloat('price')
        self.dollarToFloat('extra_people')
        self.fillNulls('cleaning_fee', '0') #Assuming missing cleaning fees indicates 0 charge.
        self.dollarToFloat('cleaning_fee')
        self.fillNulls('security_deposit', '0')
        self.dollarToFloat('security_deposit')

        #convert host response rate to float
        self.df.host_response_rate = self.df.host_response_rate.str.replace('%', '').astype(float) / 100
        
        #clean up other null vals
        self.fillNulls('bathrooms', 1.0)
        self.fillNulls('bedrooms', 1.0)
        self.fillNulls('beds', 1.0)

        self.fillNulls('review_scores_rating', self.df.review_scores_rating.mean())
        self.fillNulls('review_scores_accuracy', self.df.review_scores_accuracy.mean())
        self.fillNulls('review_scores_cleanliness', self.df.review_scores_cleanliness.mean())
        self.fillNulls('review_scores_checkin', self.df.review_scores_checkin.mean())
        self.fillNulls('review_scores_communication', self.df.review_scores_communication.mean())
        self.fillNulls('review_scores_location', self.df.review_scores_location.mean())
        self.fillNulls('review_scores_value', self.df.review_scores_value.mean())
        self.fillNulls('reviews_per_month', 0)
        
        #TODO:
        # amenities looks to be a set of possible values--need to extract
        # similarly, host_verifications
        # longitude/latitude cleanup
