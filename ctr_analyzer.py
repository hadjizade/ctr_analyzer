# -*- coding: utf-8 -*-
import sys
import os
import re
import argparse
import pandas as pd
import numpy as np

# تنظیم خروجی برای پشتیبانی از فارسی در ترمینال ویندوز
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# الگوهای تشخیص نیت کاربر (قابل ویرایش برای زبان‌ها یا موضوعات دیگر)
INTENT_PATTERNS = {
    "Commercial": r"بهترین|قیمت|خرید|ارزان|نزدیک من|تضمینی",
    "Navigational": r"دکتر\s+[آ-ی]|کلینیک\s+[آ-ی]|سایت\s+[آ-ی]",
    "Local": r"تهران|شیراز|مشهد|اصفهان|کرج|تبریز|اهواز|قم|کرمان|رشت",
    "Informational": r"چیست|علائم|علایم|درمان|عوارض|علت|چرا|چیه"
}

def to_num(s):
    """تبدیل اعداد فارسی/عربی به انگلیسی و حذف کاراکترهای اضافی"""
    if pd.isna(s): return np.nan
    s = str(s).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))
    return pd.to_numeric(s.replace(",", "").replace("%", "").strip(), errors="coerce")

def normalize_query(q):
    """نرمال‌سازی کلمات جستجو (یکسان‌سازی ی و ک، حذف علائم)"""
    q = str(q).lower().strip().replace("ي", "ی").replace("ك", "ک")
    q = re.sub(r"[^\w\s\u200c\u0600-\u06FF]", " ", q)
    return re.sub(r"\s+", " ", q).strip()

def get_intent(q):
    """تشخیص نیت کاربر بر اساس الگوهای تعریف شده"""
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            # بررسی استثناها (مثلاً اگر کلمه بهترین هم داشت، اولویت با Commercial است)
            if intent == "Navigational" and "بهترین" in q: continue
            if intent == "Local" and "بهترین" in q: continue
            return intent
    return "Generic"

def load_and_clean_data(filepath):
    """بارگذاری فایل و پاک‌سازی اولیه"""
    print(f"📂 در حال خواندن فایل: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"فایل ورودی یافت نشد: {filepath}")
        
    try:
        # تلاش برای خواندن به عنوان CSV
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except Exception:
        # اگر خطا داد، به عنوان فایل متنی (Tab-separated) می‌خوانیم
        df = pd.read_csv(filepath, sep="\t", encoding="utf-8-sig")

    df.columns = [c.strip() for c in df.columns]
    
    required_cols = ["Query", "Clicks", "Impressions", "Position"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"ستون‌های زیر در فایل شما یافت نشد: {', '.join(missing_cols)}")

    # پاک‌سازی اعداد
    df["Clicks"] = df["Clicks"].map(to_num)
    df["Impressions"] = df["Impressions"].map(to_num)
    df["Position"] = df["Position"].map(to_num)
    
    # حذف ردیف‌های خالی و فیلتر نمایش‌های کم
    df = df.dropna(subset=["Clicks", "Impressions", "Position"])
    df = df[df["Impressions"] >= 50].copy()
    
    if df.empty:
        raise ValueError("داده‌ای برای تحلیل باقی نماند (همه ردیف‌ها خالی بودند یا کمتر از ۵۰ نمایش داشتند).")

    df["ctr"] = df["Clicks"] / df["Impressions"]
    df["q_norm"] = df["Query"].map(normalize_query)
    df["wc"] = df["q_norm"].str.split().str.len()
    
    return df

def build_model(df):
    """ساخت مدل پیش‌بینی CTR چندلایه"""
    print("🧠 در حال آموزش مدل پیش‌بینی...")
    
    df["intent"] = df["q_norm"].apply(get_intent)
    df["pos_bucket"] = pd.cut(df["Position"], [1, 2, 4, 6, 11, 21, np.inf],
                              labels=["1", "2-3", "4-5", "6-10", "11-20", "20+"], right=False)
    df["wc_bucket"] = pd.cut(df["wc"], [0, 1, 2, 3, 4, 6, np.inf],
                             labels=["1", "2", "3", "4", "5-6", "7+"], right=True)

    # لایه ۱: نیت + رتبه + طول کلمه
    group_1 = df.groupby(['intent', 'pos_bucket', 'wc_bucket'], observed=False)['ctr'].median().rename('expected_ctr')
    # لایه ۲: نیت + رتبه (Fallback)
    group_2 = df.groupby(['intent', 'pos_bucket'], observed=False)['ctr'].median().rename('expected_ctr_fallback')
    # لایه ۳: فقط رتبه (Fallback نهایی)
    group_3 = df.groupby('pos_bucket', observed=False)['ctr'].median().rename('expected_ctr_base')

    df = df.merge(group_1, on=['intent', 'pos_bucket', 'wc_bucket'], how='left')
    df = df.merge(group_2, on=['intent', 'pos_bucket'], how='left')
    df = df.merge(group_3, on='pos_bucket', how='left')

    df['expected_ctr'] = df['expected_ctr'].fillna(df['expected_ctr_fallback']).fillna(df['expected_ctr_base'])
    
    df['ctr_gap'] = df['ctr'] - df['expected_ctr']
    df['ctr_gap_pct'] = (df['ctr_gap'] * 100).round(1)
    
    return df, group_2

def find_opportunities(df, min_impressions=1000, top_n=20):
    """پیدا کردن فرصت‌های طلایی (کلیک کمتر از حد انتظار)"""
    opportunities = df[df['Impressions'] >= min_impressions].copy()
    opportunities = opportunities.sort_values(by='ctr_gap', ascending=True)
    
    top_ops = opportunities.head(top_n)[['Query', 'Impressions', 'ctr', 'expected_ctr', 'ctr_gap_pct', 'Position']].copy()
    top_ops['ctr'] = (top_ops['ctr'] * 100).round(1)
    top_ops['expected_ctr'] = (top_ops['expected_ctr'] * 100).round(1)
    return top_ops

def main():
    parser = argparse.ArgumentParser(description="ابزار تحلیل و پیش‌بینی CTR سرچ کنسول")
    parser.add_argument("-i", "--input", required=True, help="مسیر فایل ورودی (TXT یا CSV)")
    parser.add_argument("-o", "--output", default="CTR_Prediction_Model.xlsx", help="مسیر فایل خروجی اکسل")
    parser.add_argument("--min-imp", type=int, default=1000, help="حداقل تعداد نمایش برای فرصت‌های طلایی (پیش‌فرض: 1000)")
    args = parser.parse_args()

    try:
        # ۱. لود و پاک‌سازی
        df = load_and_clean_data(args.input)
        
        # ۲. ساخت مدل
        df, base_model = build_model(df)
        
        # ۳. پیدا کردن فرصت‌ها
        top_opportunities = find_opportunities(df, min_impressions=args.min_imp)
        
        print("\n" + "="*80)
        print(f"🔥 {len(top_opportunities)} فرصت طلایی: صفحاتی با نمایش بالا که کمتر از حد انتظار کلیک می‌گیرند!")
        print("="*80)
        print(top_opportunities.to_string(index=False))
        
        # ۴. ذخیره خروجی
        print(f"\n💾 در حال ذخیره خروجی در: {args.output}")
        with pd.ExcelWriter(args.output, engine="openpyxl") as w:
            df.head(5000).to_excel(w, sheet_name="All_Data_Predictions", index=False)
            top_opportunities.to_excel(w, sheet_name="Golden_Opportunities", index=False)
            base_model.reset_index().to_excel(w, sheet_name="Base_Model_Table", index=False)
            
        print("✅ عملیات با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()