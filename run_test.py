import sqlite3
import os.path
import glob
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import jellyfish

class OCR_test:
    connection = None
    cursor = None
    table_name = "test"
    db_path = "test.db"
    langs = ["cn", "en", "ja", "ko", "ru"]
    #simsun.ttc
    #Arial.ttf
    font_dict = {
        "cn": "simsun.ttc",
        "en": "Geoform.ttf",
        "ja": "simsun.ttc",
        "ko": "Maplestory OTF Light.otf",
        "ru": "Geoform.ttf",
    }
    colors = ["AntiqueWhite", "Aqua", "Aquamarine", "Azure", "DeepPink", "DeepSkyBlue", "BlueViolet", "Brown", "Cornsilk", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Crimson", "Cyan", "Gold", "DarkCyan", "DarkGoldenRod", "Fuchsia", "GreenYellow", "DarkKhaki", "HotPink", "GhostWhite", "LawnGreen", "LemonChiffon", "FloralWhite", "DarkSeaGreen", "DarkSlateBlue"]
    not_model_names = ["id", "file_name"]

    def make_db(self, model_names):
        print("Creating DB")
        if os.path.isfile(self.db_path):
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print("DB already exists, skipping this step")
            return
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        sub_text = ""
        i = 0
        for model_name in model_names:
            i += 1
            name = model_name.replace("-", "_")
            sub_text += f"{name} TEXT"
            if i != len(model_names):
                sub_text += ", "
        #INSERT INTO test (`manga_ocr`) VALUES ('micro test') WHERE file_name='1'
        for lang in self.langs:
            self.cursor.execute(f"CREATE TABLE {self.table_name}_{lang} (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT NOT NULL UNIQUE, answer TEXT, {sub_text})")
        self.connection.commit()
        print("DB is created")
        

    def setup_db(self, model_names):
        for lang in self.langs:
            #Check for new model_names, if found, add one
            command = f"PRAGMA table_info(test_{lang})"
            self.cursor.execute(command)
            rows = self.cursor.fetchall()
            cur_model_names = []
            for row in rows:
                cur_model_names.append(row[1])
            for m_name in model_names:
                if not(m_name in cur_model_names):
                    command = f"ALTER TABLE test_{lang} ADD COLUMN {m_name} TEXT"
                    print(command)
                    self.cursor.execute(command)
            self.connection.commit()

            #Insert new rows with image name or ignore
            image_paths = glob.glob(f"test_data/{lang}/*")       
            for path in image_paths:
                file_name = Path(path).stem
                command = f"INSERT OR IGNORE INTO {self.table_name}_{lang} (file_name) VALUES ('{file_name}')"
                #print(command)
                self.cursor.execute(command)
            self.connection.commit()
    
    def update_db(self, model_name, predictions):
        for lang in self.langs:
            for prediction in predictions:
                file_name, pred_text, pred_lang = prediction
                pred_text = pred_text.replace("'", "''")
                if lang != pred_lang:
                    continue
                command = f"UPDATE {self.table_name}_{lang} SET {model_name} = '{pred_text}' WHERE file_name='{file_name}'"
                print(command)
                self.cursor.execute(command)
            self.connection.commit()
    def find_unOCRed_image_paths(self, model_name):
        result = []
        for lang in self.langs:
            image_paths = glob.glob(f"test_data/{lang}/*")
            for path in image_paths:
                file_name = Path(path).stem
                command =  f"SELECT {model_name} FROM {self.table_name}_{lang} WHERE file_name='{file_name}'"
                self.cursor.execute(command)
                rows = self.cursor.fetchall()
                if rows[0][0] == None or rows[0][0] == "":
                    result.append(path)
        print("RESULT")
        print(result)
        return result
    def simple_manga_ocr_test(self):
        model_name = "manga_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import manga_ocr1
        predictions = manga_ocr1.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_callisto_ocr_test(self):
        model_name = "callisto_ocr3_2b_instruct"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import callisto_ocr3_2b_instruct
        predictions = callisto_ocr3_2b_instruct.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_dddd_ocr_test(self):
        model_name = "dddd_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import dddd_ocr
        predictions = dddd_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_doctr_ocr_test(self):
        # Not works :(
        model_name = "doctr_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import doctr_ocr
        predictions = doctr_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_easy_ocr_test(self):
        model_name = "easy_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import easy_ocr
        predictions = easy_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_got_ocr_test(self):
        model_name = "got_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import got_ocr
        predictions = got_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_got_ocr_orig_test(self):
        model_name = "got_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import got_ocr_orig
        predictions = got_ocr_orig.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_granite_ocr_test(self):
        model_name = "granite_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import granite_ocr
        predictions = granite_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_internVL3_8B_test(self):
        model_name = "internVL3_8B"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import internVL3_8B
        predictions = internVL3_8B.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_internVL3_2B_test(self):
        model_name = "internVL3_2B"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import internVL3_2B
        predictions = internVL3_2B.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_olm_ocr_test(self):
        model_name = "olm_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import olm_ocr
        predictions = olm_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_paddle_ocr_test(self):
        model_name = "paddle_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import paddle_ocr
        predictions = paddle_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_paligemma_3b_gt_ocrvqa_448_test(self):
        #bad
        model_name = "paligemma_3b_gt_ocrvqa_448"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import paligemma_3b_gt_ocrvqa_448
        predictions = paligemma_3b_gt_ocrvqa_448.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_phi4_ocr_test(self):
        model_name = "phi4_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import phi4_ocr
        predictions = phi4_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_qwen2_vl_ocr_2B_instruct_test(self):
        model_name = "qwen2_vl_ocr_2B_instruct"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import qwen2_vl_ocr_2B_instruct
        predictions = qwen2_vl_ocr_2B_instruct.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_tesseract_ocr_test(self):
        model_name = "tesseract_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import tesseract_ocr
        predictions = tesseract_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_windows_ocr_test(self):
        model_name = "windows_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import windows_ocr
        predictions = windows_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_tr_ocr_test(self):
        model_name = "tr_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import tr_ocr
        predictions = tr_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_tokenized_ocr_test(self):
        #idk why i add this one
        model_name = "tokenized_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import tokenized_ocr
        predictions = tokenized_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_surya_ocr_test(self):
        model_name = "surya_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import surya_ocr
        predictions = surya_ocr.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_miniCPM_o_2_6_test(self):
        model_name = "miniCPM_o_2_6"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import miniCPM_o_2_6
        predictions = miniCPM_o_2_6.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_mini_monkey_test(self):
        model_name = "mini_monkey"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import mini_monkey
        predictions = mini_monkey.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def simple_h2ovl_mississippi_2b_test(self):
        model_name = "h2ovl_mississippi_2b"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        if len(unOCRed_image_paths) == 0:
            return
        import h2ovl_mississippi_2b
        predictions = h2ovl_mississippi_2b.test(self.langs, unOCRed_image_paths)
        self.update_db(model_name, predictions)
    def get_model_score_dict(self):
        lang = "ja"
        command = f"PRAGMA table_info(test_{lang})"
        self.cursor.execute(command)
        rows = self.cursor.fetchall()
        cur_model_names = []
        for row in rows:
            model_name = row[1]
            cur_model_names.append(model_name)
        result = {}
        for model_name in cur_model_names:
            if model_name in self.not_model_names:
                continue
            result[model_name] = 0.0
        return result
        
    
    def gen_comparison_images(self):
        text_size = 20
        
        model_score_final_dict = self.get_model_score_dict()
        for lang in self.langs:
            #Check for new model_names, if found, add one
            command = f"PRAGMA table_info(test_{lang})"
            self.cursor.execute(command)
            rows = self.cursor.fetchall()
            cur_model_names = []
            for row in rows:
                model_name = row[1]
                cur_model_names.append(model_name)
            command = f"SELECT * FROM test_{lang}"
            self.cursor.execute(command)
            rows = self.cursor.fetchall()
            width, height = 1920,1080

            model_score_lang_dict = self.get_model_score_dict()


            for row in rows:
                file_name = row[1]
                answer = row[2]
                id = row[0]
                image_path = f"test_data/{lang}/{file_name}.png"
                output_folder_path = f"test_data_compare/{lang}/"
                out_image_path = f"test_data_compare/{lang}/{file_name}.png"

                result_img = Image.new('RGBA', (width, height), 'black')
                img = Image.open(image_path).convert("RGBA")
                img_width, img_height = img.size
                result_img.paste(img, (0, 0), img)
                
                idraw = ImageDraw.Draw(result_img)
                font_name = self.font_dict[lang]
                font = ImageFont.truetype(font_name, size=text_size)
                #print(lang)
                
                cur_y = img_height + 5
                cur_x = 5
                #print("row")
                #print(row)
                for i in range(len(row)):
                    model_prediction = str(row[i])
                    
                    model_prediction = model_prediction.replace("\n", " ")
                    model_name = cur_model_names[i]
                    if model_name in self.not_model_names:
                        text = f"{model_name}: {model_prediction}"
                    else:
                        similarity = jellyfish.jaro_similarity(model_prediction, answer)
                        model_score_lang_dict[model_name] += similarity
                        similarity = round(similarity, 3)
                        text = f"{model_name}: ({similarity}){model_prediction}"
                    idraw.text((cur_x, cur_y), text, font=font, fill=self.colors[i])
                    cur_y += text_size
                
                os.makedirs(output_folder_path, exist_ok=True)
                result_img.save(out_image_path)
                #result_img.show()
                #return
            # Adding similarity score from cur lang to overall score across all langs
            for model_name, score in model_score_lang_dict.items():
                model_score_final_dict[model_name] += score
            print("Lang: " + lang)
            model_score_lang_dict = {key: val for key, val in sorted(model_score_lang_dict.items(), key = lambda ele: ele[1], reverse = True)}
            print(model_score_lang_dict)

            score_img_lang = Image.new('RGBA', (width, height), 'black')
            idraw2 = ImageDraw.Draw(score_img_lang)
            font_name = "Geoform.ttf"
            font = ImageFont.truetype(font_name, size=text_size)

            out_image_path = f"score_{lang}.png"

            cur_y = 20
            cur_x = 5
            for model_name, score in model_score_lang_dict.items():
                text = f"{model_name}: {score}"
                idraw2.text((cur_x, cur_y), text, font=font, fill=self.colors[i])
                cur_y += text_size
            score_img_lang.save(out_image_path)



        print("Final")
        model_score_final_dict = {key: val for key, val in sorted(model_score_final_dict.items(), key = lambda ele: ele[1], reverse = True)}
        print(model_score_final_dict)
        
        score_img_final = Image.new('RGBA', (width, height), 'black')
        idraw2 = ImageDraw.Draw(score_img_final)
        font_name = "Geoform.ttf"
        font = ImageFont.truetype(font_name, size=text_size)

        out_image_path = f"score_final.png"

        cur_y = 20
        cur_x = 5
        for model_name, score in model_score_final_dict.items():
            text = f"{model_name}: {score}"
            idraw2.text((cur_x, cur_y), text, font=font, fill=self.colors[i])
            cur_y += text_size
        score_img_final.save(out_image_path)

        


            
            
            
    def start(self):
        model_names = ["callisto_ocr3_2b_instruct", "dddd_ocr", "doctr_ocr", "easy_ocr", "got_ocr", "granite_ocr", "internVL3_2B", "internVL3_8B", "manga_ocr", "olm_ocr", "paddle_ocr", "paligemma_3b_gt_ocrvqa_448", "phi4_ocr", "qwen2_vl_ocr_2B_instruct", "rapid_ocr", "tesseract_ocr", "tr_ocr", "windows_ocr", "surya_ocr", "miniCPM_o_2_6", "mini_monkey", "h2ovl_mississippi_2b"]
        self.make_db(model_names)
        self.setup_db(model_names)
        
        self.gen_comparison_images()

        ##self.simple_granite_ocr_test()
        ## self.simple_callisto_ocr_test()
        ## self.simple_internVL3_8B_test()
        ##self.simple_olm_ocr_test()
        ##self.simple_paddle_ocr_test()
        #self.simple_phi4_ocr_test()
        ##self.simple_qwen2_vl_ocr_2B_instruct_test()
        ##self.simple_tesseract_ocr_test()
        ##self.simple_windows_ocr_test()
        ##self.simple_tr_ocr_test()
        ##self.simple_surya_ocr_test()
        ##self.simple_miniCPM_o_2_6_test()
        ##self.simple_mini_monkey_test()
        ##self.simple_h2ovl_mississippi_2b_test()

OCR_test().start()