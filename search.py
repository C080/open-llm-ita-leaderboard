def job_ad_web_search(title, websites):
    '''Search for job ads on the web and return a list of job descriptions'''

    from jobspy import scrape_jobs
    i = 0
    #useful_jobs_descriptions = []
    while i < 40:
        for site in websites:
            try:
                jobs = scrape_jobs(
                    site_name=[site],
                    search_term=title,
                    results_wanted=10,
                    country_indeed='USA'  # only needed for indeed / glassdoor
                )
            except Exception as e:
                print(site, e)
                i += 10  # if 403, skip to the next site
                continue

            if not jobs.empty: # if site returns jobs
                for job in jobs.itertuples():
                    if job.description is not None and len(job.description) > 100 and len(job.description) < 5000:
                        #useful_jobs_descriptions.append(job.description)
                        return job.description
                    i += 1
                    continue
            i += 10

    return "No jobs found" #if len(useful_jobs_descriptions) == 0 else useful_jobs_descriptions





if __name__ == '__main__':
    print(job_ad_web_search('technical art director', ['indeed', 'linkedin', 'zip_recruiter', 'glassdoor']))