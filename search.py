from jobspy import scrape_jobs


def web_search(title, websites):
    i = 0
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
                print(e)
                i += 10  # if 403, skip to the next site
                continue

            if not jobs.empty: # if site returns jobs
                for job in jobs.itertuples():
                    if job.description is not None:
                        return job.description
                    i += 1
                    continue
            i += 10

    return "No jobs found"



if __name__ == '__main__':
    print(web_search('technical art director', ['indeed', 'linkedin', 'zip_recruiter', 'glassdoor']))