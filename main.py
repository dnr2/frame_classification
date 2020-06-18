from fn_frame_classification import FrameNetClassifier

def run_unit_tests():
    fn_model = FrameNetClassifier()
    
    input_data = [
        ("the problem is telling which is the original document and which the copy", 68, 71),
        ("the cause of the accident is not clear", 4, 8),
        ("Rubella, also known as German measles or three-day measles, is an infection caused by the rubella virus.", 0, 6),
        ("he died after a long illness", 21, 27),
        ("for a time revolution was a strong probability", 35, 45),
    ]    
    frames_lst = [["Duplication", "Causation", "Medical_conditions", "Probability"]]
    frames_lst = frames_lst * len(input_data)
    print("\n")
    fn_model.predict_top_k_frames(input_data, k = 3)
    print("\n")
    fn_model.get_frames_probability(input_data, frames_lst)

    print("\n")
    fn_model.get_frames_probability(
        [("Glue can be made from plant or animal parts, or it can be made from oil-based chemicals.", 12, 15)],
        [["Arriving", "Building", "Cause_change", "Cooking_creation", "Creating", "Self_motion"]]
    )
    fn_model.get_frames_probability(
        [("Glue can be made from plant or animal parts, or it can be made from oil-based chemicals.", 22, 26)],
        [["Locale_by_use", "Plants"]]
    )
    fn_model.get_frames_probability(
        [("Public libraries have stories and books about lots of things.", 17, 20)],
        [["Giving_birth", "Ingest_substance", "Ingestion", "Opinion", "Possession"]]
    )
    fn_model.get_frames_probability(
        [("Male deer have antlers.", 10, 13)],
        [["Giving_birth", "Ingest_substance", "Ingestion", "Opinion", "Possession"]]
    )    
    fn_model.get_frames_probability(
        [("It is made from tomatoes, so it is sometimes called tomato sauce.", 6, 9)],
        [["Arriving", "Building", "Cause_change", "Cooking_creation", "Creating", "Intentionally_create", "Self_motion", "Manufacturing"]]
    )
    fn_model.get_frames_probability(
        [("It is made from tomatoes, so it is sometimes called tomato sauce.", 45, 50)],
        [["Being_named", "Cause_to_start", "Claim_ownership", "Communication_means", "Contacting", "Deserving", "Labeling", "Request", "Simple_naming"]]
    )

if __name__=="__main__": 
    run_unit_tests()