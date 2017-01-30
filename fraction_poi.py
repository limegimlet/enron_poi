import pickle
#from get_data import getData

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    try:
        fraction = float(poi_messages)/(all_messages)
    except:
        if poi_messages == 'NaN' or all_messages == 'NaN':
            fraction = 0
    
    return fraction

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

submit_dict = {}

for name in data_dict:
    
    data_point = data_dict[name]
    
    #fraction fr poi
    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = computeFraction (from_poi_to_this_person, to_messages)
    #print fraction_from_poi
    data_point[fraction_from_poi] = fraction_from_poi
    
    #fraction to poi 
    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = computeFraction (from_this_person_to_poi, from_messages)
    #print fraction_to_poi
    data_point[fraction_to_poi] = fraction_to_poi
    
    #fraction from vs to messages
    fraction_from_to = computeFraction(from_messages, to_messages)
    data_point[fraction_from_to] = fraction_from_to
    
    #fraction sal vs bonus
    salary = data_point['salary']
    bonus = data_point['bonus']
    fraction_sal_bonus = computeFraction(salary, bonus)
    data_point[fraction_sal_bonus] = fraction_sal_bonus
    
    #fraction exercised options vs total stock value
    '''
    options = data_point['exercised_stock_options']
    total_stock = data_point['total_stock_value']
    fraction_options_stock = computeFraction(options, total_stock)
    data_point[fraction_options_stock] = fraction_options_stock
    '''
    
    
    
    submit_dict[name] = {"fraction_to_poi": fraction_to_poi,
                         "fraction_from_poi": fraction_from_poi,
                        "fraction_sal_bonus" : fraction_sal_bonus,
                        "fraction_from_to": fraction_from_to}
    

def submitDict():
    return submit_dict                     
                      