"""Transparent, versioned DEMO rules. No bank eligibility or credit approval."""
import random

RULE_VERSION='demo-health/1.0'
def evaluate(profile,property_value=0):
    income=float(profile['income']);expenses=float(profile['expenses']);payment=float(profile['debt_payment'])
    cash=float(profile['cash']);debt=float(profile['debt']);surplus=income-expenses-payment
    dsr=payment/max(income,1);months=cash/max(expenses+payment,1)
    components={'Cash flow':max(0,min(100,100*surplus/max(income*.3,1))),
                'Debt resilience':max(0,100*(1-min(dsr/.6,1))),
                'Emergency buffer':min(100,100*months/6),
                'Evidence completeness':100 if profile['verified'] else 20}
    weights=[.35,.3,.25,.1];score=round(10*sum(v*w for v,w in zip(components.values(),weights)))
    rng=random.Random(260630);peers=[rng.randint(300,920) for _ in range(100)]
    percentile=round(sum(s<score for s in peers)/len(peers)*100)
    budget=max(0,income*.4-payment);rate=.10/12;term=240
    affordability=budget*(1-(1+rate)**(-term))/rate
    collateral=max(0,float(property_value)*.6-debt)
    capacity=min(affordability,collateral) if property_value else 0
    return {'score':score,'status':'Healthy' if score>=750 else 'Building resilience' if score>=550 else 'Needs attention',
            'components':components,'percentile':percentile,'peer_count':100,'peer_source':'Seeded synthetic cohort, not bank customers',
            'monthly_surplus':surplus,'dsr':dsr,'buffer_months':months,'illustrative_borrowing_capacity':capacity,
            'rules_version':RULE_VERSION,'income_verified':bool(profile['verified']),
            'loan_assumptions':{'annual_rate':.10,'term_months':240,'max_dsr':.4,'ltv':.6},
            'disclaimer':'DEMO rules and assumed rates only. Not a credit score, appraisal, product offer or bank approval.'}
