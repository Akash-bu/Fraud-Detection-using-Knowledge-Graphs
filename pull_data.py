import pandas as pd
from config import get_driver

driver = get_driver()


with driver.session() as session:
    result = session.run("""
        MATCH (t:Transaction)-[:HAS_DEVICE]->(d:Device)
        OPTIONAL MATCH (t)-[:USED_CARD]->(c:Card)
        OPTIONAL MATCH (t)-[:BILLED_TO]->(a:Address)
        OPTIONAL MATCH (t)-[:SENT_FROM]->(e:EmailDomain)
        RETURN t.TransactionID AS txn_id,
               t.TransactionAmt AS amount,
               t.TransactionDT AS timestamp,
               t.ProductCD AS product,
               t.isFraud AS fraud,
               t.card1 AS card,
               t.addr1 AS address,
               d.DeviceInfo AS device,
               e.P_emaildomain AS email_domain,
               c.card4 AS card_network,
               c.card6 AS card_type
    """)
    df = pd.DataFrame([r.data() for r in result])

print(df.shape)
print(df.head())
print(df['fraud'].value_counts())
print(df['product'].value_counts())
print(df[['card', 'address', 'email_domain']].nunique())
print(df[['card_network', 'card_type']].value_counts())

df.to_pickle('data/df.pkl')