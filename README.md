chs_lexis
==============================

wrangling and visualization of CHS and LexisNexis data




## Data Output Format
###  Participant level data: 

- prs_chs_ssn_altkey_2014
    - Unique identifier for each CHS participant
- prs_chs_death_2014
    - Year of death from CHS
- prs_lex_bestAddressLast_2014
    - Last_seen date of the most recent address
- prs_lex_bestAddressLastCorrect_b_2014
    - Binary indicator variable which denotes whether prs_lex_bestAddressLast_2014 occurs after the CHS date of death.  Value is 1 for no vote after death, 0 for vote after death.  No value will be present if the participant has no address listings.
- prs_lex_bestAddressLastMod_2014
    - Modified version of prs_lex_bestAddressLast_2014.  If prs_lex_bestAddressLastCorrect_b_2014 = 1, then this will be the same as prs_lex_bestAddressLast_2014.  Otherwise, it will be January 1st in the year of the CHS date of death.
- prs_lex_empRange*firstSeen_2014
    - First seen date of a range of employment.  * indicates which sequential range of employment this is. 1 <= * <= 3. Employment ranges are determined using Employment dates for each job, then eliminating overlaps by using a graph algorithm.
- prs_lex_empRange*lastSeen_2014
    - Last seen date of a range of employment.  * indicates which sequential range of employment this is. 1 <= * <= 3. Employment ranges are determined using Employment dates for each job, then eliminating overlaps by using a graph algorithm.
- prs_lex_numJobs_c_2014
    - Total number of jobs a participant has held.  Will often not match up with the number of employment intervals because of job overlap.
- prs_lex_professionalAny_b_2014
    - Binary variable indicating whether the participant has ever held a professional license
- prs_lex_professional_c_2014
    - Count of total professional licenses ever held by the participant
- prs_lex_votePrim_c_2014
    - Count of how many times the participant has voted in a primary election.
- prs_lex_voteGen_c_2014
    - Count of how many times the participant has voted in a general election.
- prs_lex_votePres_c_2014
    - Count of how many times the participant has voted in a presidential election.
- prs_lex_voteTotal_c_2014
    - Count of how many times the participant has voted in total.
- prs_lex_MostRecentVote_2014
    - Most recent date that the participant has voted.
- prs_lex_deathVote_b_2014
    - Binary indicator variable that is 1 if the most recent vote date is after the CHS date of death, and 0 if it is not.  No value will be present for participants that have never voted.
- prs_lex_propertyOwn_c_2014
    - Count of total properties that the participant has ever owned.

### Buffer level data: 

any data that **does** represent a record for each of a participants best_addresses

- adr_lex_bestAddressNum_i_2014

    - Index indicating the bestAddress number.  Creates a unique index along with ssn_altkey.

- adr_lex_bestAddressLength_2014

    - The length of time spent at the given bestAddress.  Value will be empty if there is a missing firstSeen or lastSeen date.

- adr_lex_bestAddressSameRel_c_2014

    - Number of relatives with a concordant address

- adr_lex_bestAddressSameAsso_c_2014

    - Number of associates with a concordant address



## Tests Performed

- license history
  - number of records
  - Check that lex_professional_any = 1 when lex_professional_c >= 1.
  - Check that lex_professional_any = 0 when lex_professional_c = 0.
- property history
  - number of records
- vote history
  - number of records
  - All specific types of votes should sum to lex_votetotal_c
  - lex_deathvote should only be true if year_death < lex_MostRecentDate
- work history
  - number of records
  - if the emp_range variables are **not** empty your number of jobs should be >= 1
  - if the emp_range variables  **are** empty your number of jobs should be 0
  - all first seen variables should be chronologically before the last_seen variables
- address history (participant)
  - number of records
- address history (buffer)
  - number of records
  - If an address has a length but we have NaN vals for number of relatives/associates with concordant addresses, it should be because a point wasn't able to be geocoded.  We'll match these against geocoding_success.csv from Phil.
  - If an address has NaN for length but we have vals for relatives and associates it should be because we're missing a last_seen or first_seen for that address and can't construct an interval length.  We can back-check this to the original file.


